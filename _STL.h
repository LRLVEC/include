#pragma once
#include <cstdio>
#include <_Vector.h>
#include <_String.h>
#include <_Math.h>
#include <GL/_OpenGL.h>

struct STL
{
	#pragma pack(2)
	struct Triangle
	{
		Math::vec3<float>normal;
		Math::mat3<float>vertices;
		unsigned short attrib;
		void print()const;
	};
	#pragma pack()
	String<char>name;
	unsigned int num;
	Vector<Triangle>triangles;
	STL();
	STL(String<char>const&);
	STL(STL const&);

	void printInfo()const;
};


namespace OpenGL
{
	struct STLData :Buffer<ArrayBuffer>::Data
	{
		STL stl;
		virtual void* pointer()override
		{
			return stl.triangles.data;
		}
		virtual unsigned int size()override
		{
			return sizeof(STL::Triangle)* stl.num;
		}
	};
}



inline void STL::Triangle::print() const
{
	::printf("[");
	normal.print();
	::printf(", [");
	vertices.rowVec[0].print();
	::printf(", ");
	vertices.rowVec[1].print();
	::printf(", ");
	vertices.rowVec[2].print();
	::printf("], %u]\n", attrib);
}

inline STL::STL()
	:
	name(),
	num(0),
	triangles()
{
}
inline STL::STL(String<char> const& _name)
	:
	name(_name),
	num(0),
	triangles()
{
}
inline STL::STL(STL const& a)
{
	name = a.name;
	num = a.num;
	triangles = a.triangles;
}

inline void STL::printInfo() const
{
	::printf("[");
	name.print();
	::printf(": num: %d]\n", num);
	triangles.traverse([](Triangle const& a)
		{
			a.print();
			return true;
		});
}

