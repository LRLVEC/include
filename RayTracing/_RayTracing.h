#pragma once
#include <_Vector.h>
#include <_Math.h>
#include <_File.h>
#include <GL/_OpenGL.h>

namespace RayTracing
{
	struct Model
	{
		using vec4 = Math::vec4<float>;
		using vec3 =Math::vec3<float>;
		using mat34 = Math::mat<float, 3, 4>;

		struct Color
		{
			vec4 r;
			vec4 t;
			vec3 glwo;
			float n;
		};
		struct TriangleData :OpenGL::Buffer<OpenGL::ShaderStorageBuffer>::Data
		{
			struct Triangle
			{
				mat34 vertices;
				mat34 cross;
				mat34 delta;
				vec4 sphere;
				vec4 paras;
				Color color;
			};
			Vector<Triangle>triangles;
			TriangleData()
				:
				Data(StaticDraw)
			{

			}
			TriangleData()
			{

			}
			virtual void* pointer()
			{
				return triangles.data;
			}
			virtual unsigned int size()
			{
				return sizeof(Triangle)* triangles.length;
			}
		};
		struct PlaneData :OpenGL::Buffer<OpenGL::ShaderStorageBuffer>::Data
		{
			struct Plane
			{
				vec4 paras;
				Color color;
			};
			Vector<Plane>planes;
			PlaneData()
				:
				Data(StaticDraw)
			{

			}
			virtual void* pointer()
			{
				return planes.data;
			}
			virtual unsigned int size()
			{
				return sizeof(Plane)* planes.length;
			}
		};



	};
}
