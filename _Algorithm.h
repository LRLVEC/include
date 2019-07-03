#pragma once

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
	if (p < q - 1)
	{
		T const& r(a[p]);
		int m = p, n = p;
		while (++n < q)
		{
			if (a[n] < r)
			{
				T t(a[m]);
				a[m++] = a[n];
				a[n] = t;
			}
		}
		if (p < m - 1) qsort(a, p, m);
		if (m < q - 2) qsort(a, m + 1, q);
	}
}