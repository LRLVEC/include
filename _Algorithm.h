#pragma once
#include <_Vector.h>

//TODO: add IntervalSet::operator^ ...

template<class T>inline bool judgeUp(T* const a, int p, int q)
{
	for (int c0(p); c0 < q - 1; ++c0)
		if (a[c0] > a[c0 + 1])return false;
	return true;
}

template<class T>inline void maxHeap(T* const a, int p, int q)
{
	T l((p << 1) + 1);
	T r((p + 1) << 1);
	if (a[p] < a[l])
	{
		if (r < q && a[l] < a[r])
		{
			T t(a[p]);
			a[p] = a[r];
			a[r] = t;
			if ((r + 1) << 1 < q)
				maxHeap(a, r, q);
		}
		else
		{
			T t(a[p]);
			a[p] = a[l];
			a[l] = t;
			if ((l << 1) + 1 < q)
				maxHeap(a, l, q);
		}
	}
	else
	{
		if (r < q && a[p] < a[r])
		{
			T t(a[p]);
			a[p] = a[r];
			a[r] = t;
			if ((r + 1) << 1 < q)
				maxHeap(a, r, q);
		}
	}
}
template<class T>inline void maxTopHeap(T* const a, int q)
{
	if (1 < q && *a < a[1])
	{
		if (2 < q && a[1] < a[2])
		{
			T t(*a);
			*a = a[2];
			a[2] = t;
			if (5 < q)
				maxHeap(a, 2, q);
		}
		else
		{
			T t(*a);
			*a = a[1];
			a[1] = t;
			if (3 < q)
				maxHeap(a, 1, q);
		}
	}
	else
	{
		if (2 < q && *a < a[2])
		{
			T t(*a);
			*a = a[2];
			a[2] = t;
			if (5 < q)
				maxHeap(a, 2, q);
		}
	}
}
template<class T>inline void buildMaxHeap(T* const a, int q)
{
	for (int c((q >> 1) - 1); c; --c)maxHeap(a, c, q);
	maxTopHeap(a, q);
}
template<class T>inline void heapSort(T* const a, int q)
{
	for (int c((q >> 1) - 1); c; --c)maxHeap(a, c, q);
	maxTopHeap(a, q);
	for (int c(q - 1); c; --c)
	{
		T t(a[c]);
		a[c] = a[0];
		a[0] = t;
		maxTopHeap(a, c);
	}
}

// T must has operator<, operator<=
// All sections are closed intervals.
template<class T>inline void qsort(T* const a, int p, int q)
{
	if (p + 1 < q)
	{
		T& const k(a[p]);
		int m(p + 1), n(p);
		while (++n != q)
			if (a[n] < k) { T t = a[m]; a[m++] = a[n]; a[n] = t; }
		T t = a[m - 1]; a[m - 1] = a[p]; a[p] = t;
		if (p + 2 < m)qsort(a, p, m - 1);
		if (m + 1 < n)qsort(a, m, n);
	}
}

template<class T>struct Interval
{
	static_assert(NumType<T>::value == true, "Non-numeric type not supported yet!");
	union
	{
		T data[2];
		struct
		{
			T a, b;
		};
	};

	Interval()
		:
		a(0),
		b(0)
	{
	}
	template<class R, class S>Interval(R const& _a, S const& _b)
		:
		a(_a > _b ? _b : _a),
		b(_b < _a ? _a : _b)
	{
		static_assert(NumType<R>::value == true, "Non-numeric type not supported yet!");
		static_assert(NumType<S>::value == true, "Non-numeric type not supported yet!");
	}
	template<class R, class S>Interval(R&& _a, S&& _b)
		:
		a(_a > _b ? _b : _a),
		b(_b < _a ? _a : _b)
	{
		static_assert(NumType<R>::value == true, "Non-numeric type not supported yet!");
		static_assert(NumType<S>::value == true, "Non-numeric type not supported yet!");
	}
	bool operator<(Interval<T> const& s)const
	{
		return a < s.a;
	}
	//If A^B == null, then do nothing and return this.
	Interval<T>	operator+(Interval<T>const& s)const
	{
		if ((s.a < a ? a : s.a) <= (b < s.b ? b : s.b))
		{
			return { a < s.a ? a : s.a, s.b < b ? b : s.b };
		}
		return *this;
	}
	Interval<T>& operator+=(Interval<T>const& s)
	{

		if ((s.a < a ? a : s.a) <= (b < s.b ? b : s.b))
		{
			if (s.a < a) a = s.a;
			if (b < s.b) b = s.b;
		}
		return *this;
	}
	//If A^B == null, return a invalid Interval.
	Interval<T> operator^(Interval<T>const& s)const
	{
		if ((s.a < a ? a : s.a) <= (b < s.b ? b : s.b))
		{
			return { s.a < a ? a : s.a, b < s.b ? b : s.b };
		}
		return { 1,0 };
	}
	Interval<T>& operator^=(Interval<T>const& s)
	{
		if ((s.a < a ? a : s.a) <= (b < s.b ? b : s.b))
		{
			if (a < s.a) a = s.a;
			if (s.b < b) b = s.b;
		}
		else
		{
			a = 1;
			b = 0;
		}
		return *this;
	}
	bool contains(Interval<T>const& s)const
	{
		return !((s.a < a) || (b < s.b));
	}
	bool hasIntersectionWith(Interval<T>const& s)const
	{
		return (s.a < a ? a : s.a) <= (b < s.b ? b : s.b);
	}
	bool isContainedIn(Interval<T>const& s)const
	{
		return !((a < s.a) || (s.b < b));
	}
	bool valid()const
	{
		return a <= b;
	}
	T area()const
	{
		return b - a;
	}
	void print()const;
};
template<class T>struct IntervalSet :Vector<Interval<T>>
{
	//b is the section width, which must be positive
	using B = Vector<Interval<T>>;
	IntervalSet(IntervalSet<T>const& a) :Vector<Interval<T>>(a)
	{
	}
	IntervalSet(Vector<Interval<T>>const& a) :Vector<Interval<T>>(a)
	{
	}
	template<class R>IntervalSet(Vector<R>const& a, T const& b) : Vector<Interval<T>>()
	{
		static_assert(NumType<R>::value == true, "Non-numeric type not supported yet!");
		B::malloc(a.length);
		for (int c0(0); c0 < a.length; c0++)
			B::pushBack(Interval<T>(T(a.data[c0]), T(a.data[c0]) + b));
	}
	void print()const
	{
		for (int c0(0); c0 < B::length; ++c0)B::data[c0].print();
	}
	bool checkOrder()const
	{
		int c0(0);
		while (c0 + 1 < B::length)
		{
			if (B::data[c0 + 1] < B::data[c0])return false;
			++c0;
		}
		return true;
	}
	bool checkSimplified()const
	{
		if (B::length)
		{
			if (checkOrder())
			{
				int c0(0);
				while (c0 + 1 < B::length)
				{
					if (B::data[c0].hasIntersectionWith(B::data[c0 + 1]))
						return false;
					++c0;
				}
			}
			else
			{
				IntervalSet<T>tp(*this);
				tp.sort();
				int c0(0);
				while (c0 + 1 < B::length)
				{
					if (tp.data[c0].hasIntersectionWith(tp.data[c0 + 1]))
						return false;
					++c0;
				}
			}
			return true;
		}
		return true;
	}
	T area(bool withBoder)const
	{
		if (B::length)
		{
			withBoder &= (NumType<T>::numType == IsInteger);
			T a(0);
			unsigned int n;
			if (checkSimplified())
			{
				n = B::length;
				for (int c0(0); c0 < B::length; ++c0)
					a += B::data[c0].area();
			}
			else
			{
				IntervalSet<T>tp(*this);
				tp.simplify();
				n = tp.length;
				for (int c0(0); c0 < tp.length; ++c0)
					a += tp.data[c0].area();
			}
			return a + withBoder * n;
		}
		return 0;
	}
	IntervalSet<T>& simplify()
	{
		if (B::length)
		{
			sort();
			Vector<Interval<T>>vp;
			vp.pushBack(B::data[0]);
			Interval<T>* ip(&vp.end());
			int c0(1);
			while (c0 < B::length)
			{
				if ((*ip).hasIntersectionWith(B::data[c0]))
				{
					(*ip) += B::data[c0++];
				}
				else
				{
					vp.pushBack(B::data[c0++]);
					ip = &vp.end();
				}
			}
			vp.moveTo(*this);
		}
		return *this;
	}
	IntervalSet<T>& sort()
	{
		if (!checkOrder())
			qsort(B::data, 0, B::length);
		return *this;
	}
};
