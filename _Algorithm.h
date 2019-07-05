#pragma once

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