using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using System.Collections.Immutable;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tensors;

static class EnumerableExtensions
{
    public static IEnumerable<T> SkipNth<T>(this IEnumerable<T> x, int n) =>
        x.Take(n).Concat(x.Skip(n + 1));
}

public class Tensor<T>
    where T :
    IAdditiveIdentity<T, T>,
    IAdditionOperators<T, T, T>,
    ISubtractionOperators<T, T, T>,
    IMultiplicativeIdentity<T, T>,
    IMultiplyOperators<T, T, T>,
    IDivisionOperators<T, T, T>
{
    private ImmutableArray<int> _shape;
    public ImmutableArray<int> Shape
    {
        get => _shape;
        set
        {
            int length = value.Aggregate(1, (agg, e) => agg * e);
            if (length != Length)
                throw new ArgumentException("Invalid shape!");

            _shape = value;
        }
    }

    public int Length => _data.Length;
    public int Rank => _shape.Length;

    private readonly T[] _data;

    public T this[params int[] i]
    {
        get => Get(i);
        set => Set(value, i);
    }

    public Tensor(T[] data, ImmutableArray<int> shape)
    {
        _data = data;
        Shape = shape;
    }

    public Tensor(T[] data, IEnumerable<int> shape) : this(data, shape.ToImmutableArray()) { }

    public Tensor(T[] data, ITuple shape)
    {
        _data = data;

        Span<int> shapeSpan = stackalloc int[shape.Length];
        for (int i = 0; i < shape.Length; i++)
            shapeSpan[i] = (int?)shape[i] ?? throw new ArgumentException("Invalid shape!");

        Shape = shapeSpan.ToImmutableArray();
    }

    public Tensor(Array data)
    {
        _data = new T[data.Length];
        int i = 0;
        foreach (T e in data)
        {
            _data[i] = e;
            i++;
        }

        Span<int> shapeSpan = stackalloc int[data.Rank];
        for (i = 0; i < shapeSpan.Length; i++)
            shapeSpan[i] = data.GetLength(i);
        Shape = shapeSpan.ToImmutableArray();
    }

    public T Get(ReadOnlySpan<int> indices) => _data[CalculateIndex(indices)];
    public T Set(T value, ReadOnlySpan<int> indices) => _data[CalculateIndex(indices)] = value;
    private int CalculateIndex(ReadOnlySpan<int> indices)
    {
        if (indices.Length != Rank)
            throw new ArgumentException("Invalid index!");

        int index = 0;
        for (int i = 0; i < indices.Length; i++)
        {
            if (indices[i] >= Shape[i])
                throw new ArgumentException("Index was out of bounds!");

            index *= Shape[i];
            index += indices[i];
        }
        return index;
    }

    public static Tensor<float> Random(params int[] shape)
    {
        var rand = new Random();
        int length = shape.Aggregate(1, (agg, e) => agg * e);
        float[] data = new float[length];
        for (int i = 0; i < length; i++)
            data[i] = rand.NextSingle();

        return new(data, shape);
    }

    public static Tensor<T> operator +(Tensor<T> a, Tensor<T> b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Cannot add tensors with different shapes!");

        Tensor<T> result = new(new T[a.Length], a.Shape);
        for (int i = 0; i < a._data.Length; i++)
            result._data[i] = a._data[i] + b._data[i];

        return result;
    }
    public static Tensor<T> operator -(Tensor<T> a, Tensor<T> b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Cannot subtract tensors with different shapes!");

        Tensor<T> result = new(new T[a.Length], a.Shape);
        for (int i = 0; i < a._data.Length; i++)
            result._data[i] = a._data[i] - b._data[i];

        return result;
    }
    public static Tensor<T> operator *(Tensor<T> a, Tensor<T> b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Cannot multiply tensors with different shapes!");

        Tensor<T> result = new(new T[a.Length], a.Shape);
        for (int i = 0; i < a._data.Length; i++)
            result._data[i] = a._data[i] * b._data[i];

        return result;
    }
    public static Tensor<T> operator /(Tensor<T> a, Tensor<T> b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Cannot divide tensors with different shapes!");

        Tensor<T> result = new(new T[a.Length], a.Shape);
        for (int i = 0; i < a._data.Length; i++)
            result._data[i] = a._data[i] / b._data[i];

        return result;
    }
    public static bool operator ==(Tensor<T> a, Tensor<T> b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Cannot compare tensors with different shapes!");

        for (int i = 0; i < a._data.Length; i++)
            if (!a._data[i].Equals(b._data[i]))
                return false;

        return true;
    }
    public static bool operator !=(Tensor<T> a, Tensor<T> b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Cannot compare tensors with different shapes!");

        for (int i = 0; i < a._data.Length; i++)
            if (!a._data[i].Equals(b._data[i]))
                return true;

        return false;
    }

    public Tensor<T> MatrixMultiply(Tensor<T> other, (int, int)? axes = null)
    {
        (int axis1, int axis2) = axes ?? (-1, 0);
        if (axis1 < 0)
            axis1 += Rank;
        if (axis2 < 0)
            axis2 += other.Rank;

        if (Shape[axis1] != other.Shape[axis2])
            throw new ArgumentException("Invalid axes/dimensions for multiplication!");

        var shape = Shape
            .SkipNth(axis1)
            .Concat(other.Shape.SkipNth(axis2));
        int length = shape.Aggregate(1, (agg, e) => agg * e);
        Tensor<T> result = new(new T[length], shape);

        Tensor<T> self = this; // Needed to access data of 'this' in MultiplyRecursive
        int[] selfIndex = new int[self.Rank];
        int[] otherIndex = new int[other.Rank];
        MultiplyRecursive(0, new int[result.Rank]);

        return result;

        void MultiplyRecursive(int depth, int[] indices)
        {
            if (depth == indices.Length)
            {
                T sum = T.AdditiveIdentity;
                for (int i = 0; i < self._shape[axis1]; i++)
                {
                    selfIndex[axis1] = i;
                    otherIndex[axis2] = i;
                    sum += self.Get(selfIndex) * other.Get(otherIndex);
                }
                result[indices] = sum;
            }
            else
            {
                bool loop_self = depth < self.Rank - 1;
                int shapeIndex = loop_self ? depth : depth - self.Rank + 1;
                int axis = loop_self ? axis1 : axis2;
                shapeIndex += shapeIndex >= axis ? 1 : 0; // Skip shape length at axis
                int length = loop_self ? self._shape[shapeIndex] : other._shape[shapeIndex];
                int[] index = loop_self ? selfIndex : otherIndex;
                index[shapeIndex] = 0;
                for (int i = 0; i < length; i++)
                {
                    indices[depth] = i;
                    MultiplyRecursive(depth + 1, indices);
                    index[shapeIndex]++;
                }
            }
        }
    }

    public Array ToMultidimensionalArray()
    {
        var result = Array.CreateInstance(typeof(T), Shape.ToArray());
        unsafe
        {
            Buffer.BlockCopy(_data, 0, result, 0, Length * sizeof(T));
        }
        return result;
    }

    public override bool Equals(object? obj) => base.Equals(obj);

    public override int GetHashCode() => base.GetHashCode();
}

[MemoryDiagnoser]
public class Program
{
    static readonly Tensor<float> Tensor1 = Tensor<float>.Random(4, 100, 8);
    static readonly Tensor<float> Tensor2 = Tensor<float>.Random(8, 64, 4);
    static readonly Tensor<float> Tensor3 = Tensor<float>.Random(4, 100, 8);

    public static void Main()
    {
        var t = Tensor1.MatrixMultiply(Tensor2, axes: (0, 2));
        Console.WriteLine("Shape: [" + string.Join(", ", t.Shape) + "]");
        Console.WriteLine(t[3, 2, 1, 3]);

        BenchmarkRunner.Run<Program>();
    }

    [Benchmark]
    public void Add()
    {
        var t = Tensor1 + Tensor3;
    }

    [Benchmark]
    public void Mul()
    {
        var t = Tensor1 * Tensor3;
    }

    [Benchmark]
    public void MatMul()
    {
        var t = Tensor1.MatrixMultiply(Tensor2);
    }

    [Benchmark]
    public void MatMulAxes()
    {
        var t = Tensor1.MatrixMultiply(Tensor2, axes: (0, 2));
    }
}