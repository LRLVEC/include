#pragma once
#include <RayTracing/_RayTracing.h>
#include <_File.h>
#include <_Math.h>

namespace RayTracing
{
	struct MODEL
	{
#pragma pack(1)
		using vec4 = Math::vec4<float>;
		using vec3 = Math::vec3<float>;
		using mat34 = Math::mat<float, 3, 4>;
		struct Header
		{
			unsigned int planesNum;
			unsigned int planesOffset;
			unsigned int trianglesNum;
			unsigned int trianglesOffset;
			unsigned int spheresNum;
			unsigned int spheresOffset;
			unsigned int circlesNum;
			unsigned int circlesOffset;
			unsigned int cylindersNum;
			unsigned int cylindersOffset;
			unsigned int conesNum;
			unsigned int conesOffset;
			unsigned int pointLightsNum;
			unsigned int pointLightsOffset;
		};
		struct Color
		{
			vec3 r;
			int texR;
			vec3 t;
			int texT;
			vec3 d;
			int texD;
			vec3 g;
			int texG;
			vec3 blank;
			float n;
		};
		struct PlanesData
		{
			vec4 paras;
			Color color;
		};
		struct TrianglesData
		{
			mat34 vertices;
			Color color;
		};
		struct SpheresData
		{
			vec4 sphere;
			vec4 e1;
			vec4 e2;
			Color color;
		};
		struct CirclesData
		{
			vec4 plane;
			vec3 sphere;
			float r2;
			vec4 e1;
			Color color;
		};
		struct CylindersData
		{
			vec3 c;
			float r2;
			vec3 n;
			float l;
			vec4 e1;
			Color color;
		};
		struct ConesData
		{
			vec3 c;
			float c2;
			vec3 n;
			float l2;
			vec4 e1;
			Color color;
		};
		struct PointLightsData
		{
			vec4 color;
			vec4 p;
		};
#pragma pack()
		Header header;
		PlanesData* planes;
		TrianglesData* triangles;
		SpheresData* spheres;
		CirclesData* circles;
		CylindersData* cylinders;
		ConesData* cones;
		PointLightsData* pointlights;
		MODEL() {};
		MODEL(String<char>const& _path)
		{
			FILE* temp(::fopen(_path.data, "rb+"));
			::fseek(temp, 0, SEEK_SET);
			::fread(&header, 1, sizeof(MODEL::Header), temp);
			if (header.planesNum)
			{
				::fseek(temp, header.planesOffset, SEEK_SET);
				planes = (MODEL::PlanesData*)::malloc(header.planesNum);
				for (int i = 0; i < header.planesNum; ++i)
				{
					::fread((planes + i * sizeof(MODEL::PlanesData)), 1, sizeof(MODEL::PlanesData), temp);
				}
			}
			if (header.trianglesNum)
			{
				::fseek(temp, header.trianglesOffset, SEEK_SET);
				triangles = (MODEL::TrianglesData*)::malloc(header.trianglesNum);
				for (int i = 0; i < header.trianglesNum; ++i)
				{
					::fread((triangles + i * sizeof(MODEL::TrianglesData)), 1, sizeof(MODEL::TrianglesData), temp);
				}
			}
			if (header.spheresNum)
			{
				::fseek(temp, header.spheresOffset, SEEK_SET);
				spheres = (MODEL::SpheresData*)::malloc(header.spheresNum);
				for (int i = 0; i < header.spheresNum; ++i)
				{
					::fread((spheres + i * sizeof(MODEL::SpheresData)), 1, sizeof(MODEL::SpheresData), temp);
				}
			}
			if (header.circlesNum)
			{
				::fseek(temp, header.circlesOffset, SEEK_SET);
				circles = (MODEL::CirclesData*)::malloc(header.circlesNum);
				for (int i = 0; i < header.circlesNum; ++i)
				{
					::fread((circles + i * sizeof(MODEL::CirclesData)), 1, sizeof(MODEL::CirclesData), temp);
				}
			}
			if (header.cylindersNum)
			{
				::fseek(temp, header.cylindersOffset, SEEK_SET);
				cylinders = (MODEL::CylindersData*)::malloc(header.cylindersNum);
				for (int i = 0; i < header.cylindersNum; ++i)
				{
					::fread((cylinders + i * sizeof(MODEL::cylinders)), 1, sizeof(MODEL::CylindersData), temp);
				}
			}
			if (header.planesNum)
			{
				::fseek(temp, header.planesOffset, SEEK_SET);
				planes = (MODEL::PlanesData*)::malloc(header.planesNum);
				for (int i = 0; i < header.planesNum; ++i)
				{
					::fread((planes + i * sizeof(MODEL::PlanesData)), 1, sizeof(MODEL::PlanesData), temp);
				}
			}
		}
	};
}