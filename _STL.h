#pragma once
#include <cstdio>
#include <_Vector.h>
#include <_String.h>
#include <_File.h>
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
		bool operator==(Triangle const&);
		double getMinEdgeLength()const;
		void print()const;
	};
	#pragma pack()

	String<char>name;
	Vector<Triangle>triangles;

	bool verticesUpdate;
	Vector<Math::vec3<float>>vertices;
	bool verticesRepeatedUpdate;
	Vector<Math::vec3<float>>verticesRepeated;


	STL();
	STL(String<char>const&);
	STL(STL const&);

	Vector<Math::vec3<float>>& getVertices();
	Vector<Math::vec3<float>>& getVerticesRepeated();
	void removeUseless();
	double getMinTriangleScale();
	void printInfo()const;
};

namespace OpenGL
{
	struct STLData :Buffer<ArrayBuffer>::Data
	{
		STL stl;
		virtual void* pointer()override
		{
			return stl.verticesRepeated.data;
		}
		virtual unsigned int size()override
		{
			return sizeof(Math::vec3<float>)* stl.verticesRepeated.length;
		}
		STLData() = default;
		STLData(String<char>const&);
		STLData(STL const&);
		STLData(STLData const&) = default;
	};


	inline STLData::STLData(String<char> const& _name)
		:
		stl(_name)
	{
	}
	inline STLData::STLData(STL const& _stl)
		:
		stl(_stl)
	{
	}
}

inline bool STL::Triangle::operator==(Triangle const& a)
{
	return vertices == a.vertices;
}
inline double STL::Triangle::getMinEdgeLength() const
{
	double t((vertices.rowVec[0] - vertices.rowVec[1]).square());
	double s((vertices.rowVec[1] - vertices.rowVec[2]).square());
	double k((vertices.rowVec[2] - vertices.rowVec[0]).square());
	double n(t <= k ? t : k);
	return (n <= s ? n : s);
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
	triangles(),
	verticesUpdate(false),
	vertices(),
	verticesRepeatedUpdate(false),
	verticesRepeated()
{
}
inline STL::STL(String<char> const& _name)
	:
	name(_name),
	triangles(),
	verticesUpdate(false),
	vertices(),
	verticesRepeatedUpdate(false),
	verticesRepeated()
{
}
inline STL::STL(STL const& a)
	:
	name(a.name),
	triangles(a.triangles),
	verticesUpdate(a.verticesUpdate),
	vertices(a.vertices),
	verticesRepeatedUpdate(a.verticesRepeatedUpdate),
	verticesRepeated(a.verticesRepeated)
{
}



inline Vector<Math::vec3<float>>& STL::getVertices()
{

}
inline Vector<Math::vec3<float>>& STL::getVerticesRepeated()
{
	verticesRepeated.malloc(triangles.length * 3);
	verticesRepeated.length = 3 * triangles.length;
	for (int c0(0); c0 < triangles.length; ++c0)
	{
		verticesRepeated.data[3 * c0] = triangles.data[c0].vertices.rowVec[0];
		verticesRepeated.data[3 * c0 + 1] = triangles.data[c0].vertices.rowVec[1];
		verticesRepeated.data[3 * c0 + 2] = triangles.data[c0].vertices.rowVec[2];
	}
	return verticesRepeated;
}
inline void STL::removeUseless()
{
	for (int c0(0); c0 < triangles.length; ++c0)
		if (triangles[c0].getMinEdgeLength() == 0.0)
			triangles.omit(c0--);
}
inline double STL::getMinTriangleScale()
{
	if (!triangles.length)return 0.0;
	double t(triangles[0].getMinEdgeLength());
	for (int c0(1); c0 < triangles.length; ++c0)
	{
		double s(triangles[c0].getMinEdgeLength());
		if (t > s)t = s;
	}
	return sqrt(t);
}
inline void STL::printInfo() const
{
	::printf("[");
	name.print();
	::printf(": num: %d]\n", triangles.length);
	triangles.traverse([](Triangle const& a)
		{
			a.print();
			return true;
		});
}

//File...
inline File& File::createSTL(String<char>const& _name, STL const& _stl)
{
	FILE* temp(::fopen((property.path + _name).data, "wb+"));
	::fwrite(_stl.name, 1, _stl.name.length + 1, temp);
	::fseek(temp, 80, SEEK_SET);
	::fwrite(&_stl.triangles.length, 4, 1, temp);
	::fwrite(_stl.triangles.data, 1, _stl.triangles.length * 50, temp);
	::fclose(temp);
	build();
	return *this;
}
inline STL File::readSTL() const
{
	if (!this)return String<char>();
	FILE* temp(::fopen((property.path + property.file.name).data, "rb+"));
	::fseek(temp, 0, SEEK_SET);
	char t[100];
	::fread(t, 1, 80, temp);
	STL r(t);
	unsigned int _num;
	::fread(&_num, 4, 1, temp);
	int c(1);
	while (c < _num)c <<= 1;
	r.triangles.data = (STL::Triangle*)::malloc(sizeof(STL::Triangle) * c);
	r.triangles.length = _num;
	r.triangles.lengthAll = c;
	::fread(r.triangles.data, sizeof(STL::Triangle), _num, temp);
	::fclose(temp);
	return r;
}
inline STL File::readSTL(String<char> const& _name) const
{
	if (!this)return String<char>();
	FILE* temp(::fopen((property.path + _name).data, "rb+"));
	::fseek(temp, 0, SEEK_SET);
	char t[100];
	::fread(t, 1, 80, temp);
	STL r(t);
	unsigned int _num;
	::fread(&_num, 4, 1, temp);
	int c(1);
	while (c < _num)c <<= 1;
	r.triangles.data = (STL::Triangle*)::malloc(sizeof(STL::Triangle) * c);
	r.triangles.length = _num;
	r.triangles.lengthAll = c;
	::printf("%d\n", ::fread(r.triangles.data, sizeof(STL::Triangle), _num, temp));
	::printf("%d\n", ::ftell(temp));
	::fclose(temp);
	return r;
}

