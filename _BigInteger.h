#pragma once
#include <_String.h>
#include <random>
#include <complex>
#include <_AVX2.h>
#include <_Time.h>

//Big non-negative integer
struct BigInt
{
	static constexpr double Pi = 3.14159265358979323846264338327950288L;
	static constexpr double E = 2.71828182845904523536028747135266250L;
	static constexpr unsigned long long Base = 1000000000LLU;
	static constexpr unsigned long long M = 3221225473LLU;
	//         m = r * 2 ^ k + 1, g
	//3221225473   3       30     5
	struct Data
	{
		unsigned int* pos;
		unsigned int length;
		bool initialized;
		Data() :pos(nullptr), length(0), initialized(false) {}
		Data(unsigned int _length)
			:
			pos((unsigned int*)AVX2::mallocAVX2(_length * sizeof(unsigned int))),
			length(_length),
			initialized(true)
		{
			Timer timer;
			timer.begin();
			unsigned long offset;
			_BitScanForward(&offset, _length);
			offset = 32 - offset;
			if (_length >= 8)
			{
				__m256i x, y, z, delta(_mm256_set1_epi32(8));
				for (unsigned int c0(0); c0 < 8; ++c0)
					z.m256i_u32[c0] = c0;
				__m256i* p((__m256i*)pos);
				__m256i r0(_mm256_set1_epi32(0x0f0f0f0f)), r1, r2, r3;
				r1.m256i_u64[0] = 0x0e060a020c040800;
				r1.m256i_u64[1] = 0x0f070b030d050901;
				r1.m256i_u64[2] = 0x0e060a020c040800;
				r1.m256i_u64[3] = 0x0f070b030d050901;
				r2.m256i_u64[0] = 0xe060a020c0408000;
				r2.m256i_u64[1] = 0xf070b030d0509010;
				r2.m256i_u64[2] = 0xe060a020c0408000;
				r2.m256i_u64[3] = 0xf070b030d0509010;
				r3.m256i_u64[0] = 0x0405060700010203;
				r3.m256i_u64[1] = 0x0c0d0e0f08090a0b;
				r3.m256i_u64[2] = 0x0405060700010203;
				r3.m256i_u64[3] = 0x0c0d0e0f08090a0b;
				do
				{
					x = z;
					y = _mm256_and_si256(r0, x);
					x = _mm256_andnot_si256(r0, x);
					x = _mm256_srli_epi32(x, 4);
					y = _mm256_shuffle_epi8(r2, y);
					x = _mm256_shuffle_epi8(r1, x);
					x = _mm256_or_si256(x, y);
					x = _mm256_shuffle_epi8(x, r3);
					x = _mm256_srli_epi32(x, offset);
					*p++ = x;
					z = _mm256_add_epi32(z, delta);
				} while (_length -= 8);
			}
			else
			{
				for (unsigned long long c0(0); c0 < _length; ++c0)
				{
					unsigned int x(c0);
					x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
					x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
					x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
					x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
					pos[c0] = ((x >> 16) | (x << 16)) >> offset;
				}
			}
			timer.end();
			timer.print("Init pos: ");
		}
		Data(Data const& a)
			:
			pos(nullptr),
			length(0),
			initialized(false)
		{
			if (a.pos)
			{
				pos = (unsigned int*)AVX2::mallocAVX2(a.length * sizeof(unsigned int));
				length = a.length;
				memcpy(pos, a.pos, a.length * sizeof(unsigned int));
				initialized = true;
			}
		}
		Data(Data&& a)
			:
			pos(a.pos),
			length(a.length),
			initialized(a.initialized)
		{
			a.pos = nullptr;
			a.length = 0;
			a.initialized = false;
		}
		Data& operator=(Data&& a)noexcept
		{
			if (!initialized)
			{
				pos = a.pos;
				length = a.length;
				initialized = true;
				a.pos = nullptr;
				a.length = 0;
				a.initialized = false;
			}
			return *this;
		}
		~Data()
		{
			if (pos)AVX2::freeAVX2(pos);
			length = 0;
			initialized = false;
		}
	};

	static Vector<Data> cache;
	static bool initialized;

	unsigned long long* data;
	unsigned long long length;
	unsigned long long lengthAll;

	BigInt()
		:
		data(nullptr),
		length(0),
		lengthAll(0)
	{
	}
	BigInt(unsigned long long a)
		:BigInt()
	{
		if (a >= Base)length = lengthAll = 2;
		else length = lengthAll = 1;
		data = (unsigned long long*)::malloc(lengthAll * sizeof(unsigned long long));
		data[0] = a % Base;
		if (a >= Base)data[1] = a / Base;
	}
	BigInt(unsigned long long _length, std::mt19937_64& mt)
		:
		data((unsigned long long*)::malloc(_length * sizeof(unsigned long long))),
		length(_length),
		lengthAll(_length)
	{
		std::uniform_int_distribution<unsigned int> rd(0, 999999999);
		for (unsigned long long c0(0); c0 < _length; ++c0)
			data[c0] = rd(mt);
	}
	BigInt(String<char>const& str)
		:BigInt()
	{
		if (checkBigInt(str))
		{
			length = lengthAll = (str.length + 8) / 9;
			data = (unsigned long long*)::malloc(lengthAll * sizeof(unsigned long long));
			unsigned long long pos(length - 1);
			unsigned long long md(str.length % 9);
			if (md)
			{
				unsigned long long tp(str.data[0] - unsigned long long('0'));
				for (unsigned long long c0(1); c0 < md; ++c0)
				{
					tp *= 10;
					tp += str.data[c0] - unsigned long long('0');
				}
				data[length - 1] = tp;
			}
			if (length - (md ? 1 : 0))
			{
				unsigned long long ending(length - (md ? 2 : 1));
				for (unsigned long long c0(0); c0 <= ending; ++c0)
				{
					unsigned long long tp(str.data[c0 * 9 + md] - unsigned long long('0'));
					for (long long c1(1); c1 < 9; ++c1)
					{
						tp *= 10;
						tp += str.data[(c0 * 9 + md) + c1] - unsigned long long('0');
					}
					data[ending - c0] = tp;
				}
			}
		}
	}
	BigInt(BigInt&& a)noexcept
		:
		data(a.data),
		length(a.length),
		lengthAll(a.lengthAll)
	{
		a.data = nullptr;
		a.length = a.lengthAll = 0;
	}
	BigInt(unsigned long long* _data, unsigned long long _length, unsigned long long _lengthAll)
		:
		data(_data),
		length(_length),
		lengthAll(_lengthAll)
	{
	}
	~BigInt()
	{
		if (data)
		{
			::free(data);
			data = nullptr;
		}
		length = lengthAll = 0;
	}
	//initialize cache
	static void init()
	{
		if (!initialized)
		{
			//effective: 2^1 - 2^30
			cache.malloc(31);
			initialized = true;
		}
	}
	static void init(unsigned _pow)
	{
		if (!initialized)
		{
			cache.malloc(31);
			initialized = true;
		}
		if (_pow && _pow < 31)
			if (!cache.data[_pow].initialized)
				cache.data[_pow] = Data(1 << _pow);
	}
	//malloc
	BigInt& malloc(unsigned long long _length)
	{
		if (!_length)return *this;
		if (length + _length > lengthAll)
		{
			if (!lengthAll)lengthAll = 1;
			while (length + _length > lengthAll)lengthAll <<= 1;
			unsigned long long* tp = (unsigned long long*)::malloc(lengthAll * sizeof(unsigned long long));
			::memcpy(tp, data, length * sizeof(unsigned long long));
			::memset(tp + length, 0, (lengthAll - length) * sizeof(unsigned long long));
			free(data);
			data = tp;
		}
		return *this;
	}
	//realloc
	BigInt& realloc(unsigned long long _length)
	{
		if (data)
		{
			if (_length < lengthAll)
				data = (unsigned long long*)::realloc(data, _length * sizeof(unsigned long long));
		}
		else
			data = (unsigned long long*)::malloc(_length * sizeof(unsigned long long));
		return *this;
	}
	//check if a String is a big integer
	static bool checkBigInt(String<char>const& str)
	{
		if (str.length == 0)return false;
		for (unsigned long long c0(0); c0 < str.length; ++c0)
		{
			char p(str.data[c0]);
			if (p < '0' || p > '9')
				return false;
		}
		return true;
	}

	//>, <, >=, <=, ==
	bool operator>(BigInt const& b)const
	{
		if (length > b.length)return true;
		else if (length < b.length)return false;
		else if (length)
		{
			unsigned int c0(length - 1);
			while (c0)
			{
				if (data[c0] > b.data[c0])return true;
				else if (data[c0] < b.data[c0])return false;
				else --c0;
			}
			if (data[c0] > b.data[c0])return true;
		}
		return false;
	}
	bool operator<(BigInt const& b)const
	{
		if (length < b.length)return true;
		else if (length > b.length)return false;
		else if (length)
		{
			unsigned int c0(length - 1);
			while (c0)
			{
				if (data[c0] < b.data[c0])return true;
				else if (data[c0] > b.data[c0])return false;
				else --c0;
			}
			if (data[c0] < b.data[c0])return true;
		}
		return false;
	}
	bool operator>=(BigInt const& b)const
	{
		if (length > b.length)return true;
		else if (length < b.length)return false;
		else if (length)
		{
			unsigned int c0(length - 1);
			while (c0)
			{
				if (data[c0] > b.data[c0])return true;
				else if (data[c0] < b.data[c0])return false;
				else --c0;
			}
			if (data[c0] >= b.data[c0])return true;
		}
		return false;
	}
	bool operator<=(BigInt const& b)const
	{
		if (length < b.length)return true;
		else if (length > b.length)return false;
		else if (length)
		{
			unsigned int c0(length - 1);
			while (c0)
			{
				if (data[c0] < b.data[c0])return true;
				else if (data[c0] > b.data[c0])return false;
				else --c0;
			}
			if (data[c0] <= b.data[c0])return true;
		}
		return false;
	}
	bool operator==(BigInt const& b)const
	{
		if (length == b.length)
		{
			if (length)
				return !memcmp(data, b.data, length * sizeof(unsigned long long));
			return true;
		}
		return false;
	}
	//add: this + b -> c
	BigInt& add(BigInt const& b, BigInt& c)const
	{
		//must not be null BigInt!
		if (length && b.length)
		{
			unsigned long long maxl, minl;
			if (length >= b.length)
			{
				maxl = length;
				minl = b.length;
			}
			else
			{
				maxl = b.length;
				minl = length;
			}
			if (c.lengthAll < maxl + 1)
				c.malloc(maxl + 1 - c.lengthAll);
			unsigned long long carry(data[0] + b.data[0]);
			c.data[0] = carry % Base;
			carry /= Base;
			unsigned int c0(1);
			for (; c0 < minl; ++c0)
			{
				carry += data[c0] + b.data[c0];
				c.data[c0] = carry % Base;
				carry /= Base;
			}
			unsigned long long* left(length == maxl ? data : b.data);
			if (minl < maxl && carry)
			{
				for (; c0 < maxl; ++c0)
				{
					if (carry == 0)break;
					carry += left[c0];
					c.data[c0] = carry % Base;
					carry /= Base;
				}
			}
			if (c0 < maxl)
				::memcpy(c.data + c0, left + c0, (maxl - c0) * sizeof(unsigned long long));
			if (carry)
			{
				c.length = maxl + 1;
				c.data[maxl] = carry;
			}
			else c.length = maxl;
		}
		return c;
	}
	//minus: this - b -> c(return abs(this - b))
	BigInt& minus(BigInt const& b, BigInt& c)const
	{
		if (length && b.length)
		{
			unsigned long long* bigger, * smaller;
			unsigned long long maxl, minl;
			if (operator>=(b))
			{
				bigger = data;
				smaller = b.data;
				maxl = length;
				minl = b.length;
			}
			else
			{
				bigger = b.data;
				smaller = data;
				maxl = b.length;
				minl = length;
			}
			if (c.lengthAll < maxl)
				c.malloc(maxl - c.lengthAll);
			long long borrow(long long(bigger[0]) - long long(smaller[0]));
			c.data[0] = (borrow + Base) % Base;
			borrow >>= 63;
			unsigned long long c0(1);
			for (; c0 < minl; ++c0)
			{
				borrow += long long(bigger[c0]) - long long(smaller[c0]);
				c.data[c0] = (borrow + Base) % Base;
				borrow >>= 63;
			}
			if (minl < maxl && borrow)
			{
				for (; c0 < maxl; ++c0)
				{
					if (borrow == 0)break;
					borrow += long long(bigger[c0]);
					c.data[c0] = (borrow + Base) % Base;
					borrow >>= 63;
				}
			}
			if (c0 < maxl)
				::memcpy(c.data + c0, bigger + c0, (maxl - c0) * sizeof(unsigned long long));
			while (c.data[--maxl] == 0)
				if (maxl == 0)break;
			c.length = maxl + 1;
		}
		return c;
	}
	//mult: this * b -> c
	void ntt(unsigned long long* a, unsigned long long _length)
	{

	}
	BigInt& mult(BigInt const& b, BigInt& c, BigInt& tpA, BigInt& tpB, BigInt& tpC)const
	{
		if (length && b.length)
		{
			unsigned long long maxl(length + b.length);
			unsigned long long total;
			unsigned long n;
			_BitScanReverse(&n, maxl);
			if (maxl > (1llu << n))n++;
			total = 1llu << n;
			unsigned long long* dataC;//temp data
			if (tpA.lengthAll < total)
				tpA.realloc(total);
			if (tpB.lengthAll < total)
				tpB.realloc(total);
			if (c.lengthAll < total)
			{
				if (tpC.lengthAll < total)
				{
					//doesn't use tpC
					c.realloc(total);
					dataC = c.data;
				}
				else
				{
					c.realloc(maxl);
					dataC = tpC.data;
				}
			}
			init(n);
		}
	}

	//+-
	BigInt operator+(BigInt const& b)const
	{
		BigInt r;
		add(b, r);
		return r;
	}
	BigInt operator-(BigInt const& b)const
	{
		BigInt r;
		minus(b, r);
		return r;
	}

	//random integer (Base: 1000000000)
	BigInt& random(unsigned long long _length, std::mt19937_64& mt)
	{
		std::uniform_int_distribution<unsigned int> rd(0, 999999999);
		if (lengthAll < _length)
		{
			::free(data);
			data = (unsigned long long*)::malloc(_length * sizeof(unsigned long long));
			lengthAll = _length;
		}
		length = _length;
		for (unsigned long long c0(0); c0 < _length; ++c0)
			data[c0] = rd(mt);
		return *this;
	}

	//print
	void print()const
	{
		::printf("%llu", data[length - 1]);
		if (length > 1)
		{
			for (unsigned long long c0(length - 2); c0 > 0; --c0)
				::printf("%09llu", data[c0]);
			::printf("%09llu", data[0]);
		}
		::printf("\n");
	}
	void printInfo()const
	{
		unsigned long long digits(0);
		if (length)
		{
			digits = 1;
			unsigned long long n(length - 1);
			while (!data[n])
			{
				if (n == 0)break;
				n--;
			}
			if (data[n])
				digits += log10(data[length - 1]) + n * 9;
		}
		::printf("[Digits: %llu]\n", digits);
	}
};

Vector<BigInt::Data> BigInt::cache = Vector<BigInt::Data>();
bool BigInt::initialized = false;