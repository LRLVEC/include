#pragma once
#include <new>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <initializer_list>
#include <_TemplateMeta.h>

template<class T>struct Vector
{
	using elementType = T;
	T* data;
	int length;
	int lengthAll;
	//Construction
	Vector();
	Vector(std::initializer_list<T>const&);
	Vector(Vector<T>const&);
	template<class R>Vector(Vector<R>const&);
	Vector(T const&);
	//Destruction
	~Vector();
	//opetrator=
	Vector<T>& operator=	(Vector<T>&&);
	Vector<T>& operator=	(const Vector<T>&);
	//opetrator+
	Vector<T>	operator+	(const Vector<T>&);
	Vector<T>& operator+=	(const Vector<T>&);
	//malloc
	Vector<T>& malloc(unsigned int);
	//element
	T& begin();
	T& end();
	T* endptr();
	T& operator[](unsigned int);
	//Find position
	int posFirst(T const&)const;
	Vector<unsigned int>posAll(T const&)const;
	//Find element
	template<class R>T& findFirst(R const&);
	template<class R>T& findFirst(bool(*cmp)(T const&, R const&), R const&);
	T& findFirst(T const&);
	Vector<T*> find(T const&);

	//add, inverse...
	Vector& pushBack();
	Vector& pushBack(const T&);
	Vector& popBack();
	Vector& insert(T&&, unsigned int);
	Vector& inverse();
	Vector& omit(unsigned int);
	//traverse
	bool traverse(bool(*p)(T&));
	bool traverse(bool(*p)(T const&))const;
};
//Construction
template<class T>inline Vector<T>::Vector()
{
	data = nullptr;
	lengthAll = length = 0;
}
template<class T>inline Vector<T>::Vector(std::initializer_list<T>const& a)
{
	length = (int)a.size();
	lengthAll = 1;
	while (lengthAll < length)
		lengthAll <<= 1;
	data = (T*)std::malloc(lengthAll * sizeof(T));
	for (int c1 = 0; c1 < length; c1++)
		new(data + c1)T(*(a.begin() + c1));
}
template<class T>inline Vector<T>::Vector(Vector<T>const& a)
{
	if (this == &a)return;
	lengthAll = a.lengthAll;
	length = a.length;
	data = (T*)std::malloc(a.lengthAll * sizeof(T));
	for (int c1 = 0; c1 < length; c1++)
		new(data + c1)T(a.data[c1]);
}
template<class T>template<class R>inline Vector<T>::Vector(Vector<R>const& a)
{
	lengthAll = a.lengthAll;
	length = a.length;
	data = (T*)std::malloc(a.lengthAll * sizeof(T));
	for (int c1 = 0; c1 < length; c1++)
		new(data + c1)T(a.data[c1]);
}
template<class T>inline Vector<T>::Vector(T const& a)
	:
	data((T*)::malloc(2 * sizeof(T))),
	length(1),
	lengthAll(2)
{
	new(data)T(a);
}
//Destruction
template<class T>inline Vector<T>::~Vector()
{
	if (data)
	{
		for (int c1 = 0; c1 < length; c1++)(data + c1)->~T();
		free(data);
		data = nullptr;
	}
}
//opetrator=
template<class T>inline Vector<T>& Vector<T>::operator= (Vector<T>&& a)
{
	if (this == &a)return *this;
	lengthAll = a.lengthAll;
	length = a.length;
	data = (T*)std::malloc(a.lengthAll * sizeof(T));
	for (int c1 = 0; c1 < length; c1++)
		new(data + c1)T(a.data[c1]);
	return *this;
}
template<class T>inline Vector<T>& Vector<T>::operator= (Vector<T>const& a)
{
	if (this == &a)return *this;
	this->~Vector();
	lengthAll = a.lengthAll;
	length = a.length;
	data = (T*)std::malloc(a.lengthAll * sizeof(T));
	for (int c1 = 0; c1 < length; c1++)
		new(data + c1)T(a.data[c1]);
	return *this;
}
//operator+
template<class T>inline Vector<T>	Vector<T>::operator+ (Vector<T>const& a)
{
	Vector<T>* tp = new Vector<T>;
	tp->lengthAll = 1 << (1 + (int)log2(tp->length = length + a.length));
	tp->data = (T*)std::malloc(tp->lengthAll * sizeof(T));
	for (int c1 = 0; c1 < length; c1++)
		new(tp->data + c1)T(data[c1]);
	for (int c1 = 0; c1 < a.length; c1++)
		new(tp->data + c1 + length)T(a.data[c1]);
	return *tp;
}
template<class T>inline Vector<T>& Vector<T>::operator+=(Vector<T>const& a)
{
	if (length + a.length <= lengthAll)
	{
		for (int c1 = 0; c1 < a.length; c1++)
			new(data + c1 + length)T(a.data[c1]);
		length += a.length;
		return *this;
	}
	lengthAll = 1 << (1 + (int)log2(length + a.length));
	T* tp = (T*)std::malloc(lengthAll * sizeof(T));
	for (int c1 = 0; c1 < length; c1++)
		new(tp + c1)T(data[c1]);
	for (int c1 = 0; c1 < a.length; c1++)
		new(tp + c1 + length)T(a.data[c1]);
	for (int c1 = 0; c1 < length; c1++)
		(data + c1)->~T();
	free(data);
	data = tp;
	length += a.length;
	return *this;
}
//malloc
template<class T>inline Vector<T>& Vector<T>::malloc(unsigned int a)
{
	if (!a)return *this;
	if (length + a > lengthAll)
	{
		if (!lengthAll)lengthAll = 1;
		while (length + a > lengthAll)lengthAll <<= 1;
		T* tp = (T*)std::malloc(lengthAll * sizeof(T));
		for (int c1 = 0; c1 < length; c1++)
		{
			new(tp + c1)T(data[c1]);
			(data + c1)->~T();
		}
		length += a;
		free(data);
		data = tp;
		return *this;
	}
	else return *this;
}
//element
template<class T>inline T& Vector<T>::begin()
{
	return data[0];
}
template<class T>inline T& Vector<T>::end()
{
	return data[length - 1];
}
template<class T>inline T* Vector<T>::endptr()
{
	return data + length - 1;
}
template<class T>inline T& Vector<T>::operator[](unsigned int a)
{
	return data[a];
}
//Find position
template<class T>inline int Vector<T>::posFirst(T const& a)const
{
	for (int c1 = 0; c1 < length; c1++)
		if (data[c1] == a)return c1;
	return -1;
}
template<class T>inline Vector<unsigned int> Vector<T>::posAll(T const& a) const
{
	Vector<unsigned int>r;
	for (int c1(0); c1 < length; c1++)
		if (data[c1] == a)r.pushBack(c1);
	return r;
}
//Find element
template<class T>template<class R>inline T& Vector<T>::findFirst(R const& a)
{
	for (int c1 = 0; c1 < length; c1++)
		if (data[c1] == a)return data[c1];
	return *(T*)NULL;
}
template<class T>template<class R>inline T& Vector<T>::findFirst(bool(*cmp)(T const&, R const&), R const& a)
{
	for (int c1 = 0; c1 < length; c1++)
		if (cmp(data[c1], a))return data[c1];
	return *(T*)NULL;
}
template<class T>inline T& Vector<T>::findFirst(T const& a)
{
	for (int c1 = 0; c1 < length; c1++)
		if (data[c1] == a)return data[c1];
	return *(T*)NULL;
}
template<class T>inline Vector<T*> Vector<T>::find(T const& a)
{
	Vector<T*>r;
	for (int c1(0); c1 < length; c1++)
		if (data[c1] == a)r.pushBack(data + c1);
	return r;

}
//add...
template<class T>inline Vector<T>& Vector<T>::pushBack()
{
	if (lengthAll == length)
	{
		lengthAll = (lengthAll ? (lengthAll << 1) : 1);
		T* tp = (T*)std::malloc(lengthAll * sizeof(T));
		for (int c1 = 0; c1 < length; c1++)
		{
			new(tp + c1)T(data[c1]);
			(data + c1)->~T();
		}
		free(data);
		data = tp;
	}
	new(data + length++)T();
	return *this;
}
template<class T>inline Vector<T>& Vector<T>::pushBack(T const& a)
{
	if (lengthAll == length)
	{
		lengthAll = (lengthAll ? (lengthAll << 1) : 1);
		T* tp = (T*)std::malloc(lengthAll * sizeof(T));
		if (&a >= data && &a < data + length)
		{
			T temp(a);
			for (int c1 = 0; c1 < length; c1++)
			{
				new(tp + c1)T(data[c1]);
				(data + c1)->~T();
			}
			free(data);
			data = tp;
			new(data + length++)T(temp);
		}
		else
		{
			for (int c1 = 0; c1 < length; c1++)
			{
				new(tp + c1)T(data[c1]);
				(data + c1)->~T();
			}
			free(data);
			data = tp;
			new(data + length++)T(a);
		}
	}
	else
		new(data + length++)T(a);
	return *this;
}
template<class T>inline Vector<T>& Vector<T>::popBack()
{
	if (--length <= (lengthAll >> 2))
	{
		lengthAll >>= 1;
		T* tp = (T*)std::malloc(lengthAll * sizeof(T));
		for (int c1 = 0; c1 < length; c1++)
		{
			new(tp + c1)T(data[c1]);
			(data + c1)->~T();
		}
		(data + length)->~T();
		free(data);
		data = tp;
	}
	else
		(data + length)->~T();
	return *this;
}
template<class T>inline Vector<T>& Vector<T>::insert(T&& a, unsigned int b)
{
	if (b == length)return pushBack(a);
	if (lengthAll == length)
	{
		lengthAll = (lengthAll ? (lengthAll << 1) : 1);
		T* tp = (T*)std::malloc(lengthAll * sizeof(T));
		for (int c1 = 0; c1 < b; c1++)
		{
			new(tp + c1)T(data[c1]);
			(data + c1)->~T();
		}
		for (int c1 = b + 1; c1 <= length; c1++)
		{
			new(tp + c1)T(data[c1]);
			(data + c1)->~T();
		}
		free(data);
		data = tp;
	}
	else
	{
		for (int c1 = length - 1; c1 >= b; c1--)
		{
			new(data + c1 + 1)T(data[c1]);
			(data + c1)->~T();
		}
	}
	length++;
	new(data + b)T(a);
	return *this;
}
template<class T>inline Vector<T>& Vector<T>::inverse()
{
	//No exchange optimization yet
	if (length == 0 || length == 1)return *this;
	T* tp = (T*)std::malloc(lengthAll * sizeof(T));
	for (int c0(0); c0 < length / 2; ++c0)
	{
		new(tp)T(data[c0]);
		(data + c0)->~T();
		new(data + c0)T(data[length - c0 - 1]);
		(data + length - c0 - 1)->~T();
		new(data + length - c0 - 1)T(tp);
		tp->~T();
	}
	::free(tp);
	return *this;
}
template<class T>inline Vector<T>& Vector<T>::omit(unsigned int b)
{
	if (b == length - 1)
		return popBack();
	if (--length <= (lengthAll >> 2))
	{
		lengthAll >>= 1;
	}
	T* tp = (T*)std::malloc(lengthAll * sizeof(T));
	for (int c1 = 0; c1 < b; c1++)
	{
		new(tp + c1)T(data[c1]);
		(data + c1)->~T();
	}
	(data + b)->~T();
	for (int c1 = b + 1; c1 <= length; c1++)
	{
		new(tp + c1 - 1)T(data[c1]);
		(data + c1)->~T();
	}
	free(data);
	data = tp;
	return *this;
}
//traverse
template<class T>inline bool Vector<T>::traverse(bool(*p)(T&))
{
	for (int c1 = 0; c1 < length; c1++)
		if (!p(data[c1]))return false;
	return true;
}
template<class T>inline bool Vector<T>::traverse(bool(*p)(T const&))const
{
	for (int c1 = 0; c1 < length; c1++)
		if (!p(data[c1]))return false;
	return true;
}
