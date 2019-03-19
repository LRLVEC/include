#pragma once
#include <_Vector.h>
#include <_Math.h>
#include <_File.h>
#include <GL/_OpenGL.h>

namespace RayTracing
{
	struct View :OpenGL::Buffer::Data
	{
		Math::mat<float, 4, 2>vertices;
		View()
			:
			Data(StaticDraw),
			vertices
			({ {-1.0f,-1.0f},
				{1.0f,-1.0f},
				{1.0f,1.0f},
				{-1.0f,1.0f} })
		{
		}
		virtual void* pointer()override
		{
			return vertices.array;
		}
		virtual unsigned int size()override
		{
			return sizeof(vertices);
		}
	};
	struct FrameScale :OpenGL::Buffer::Data
	{
		Math::vec2<unsigned int>scale;
		FrameScale(Math::vec2<unsigned int>const& _scale)
			:
			Data(StaticDraw),
			scale(_scale)
		{
		}
		virtual void* pointer()override
		{
			return scale.data;
		}
		virtual unsigned int size()override
		{
			return sizeof(scale);
		}
	};
	struct FrameData :OpenGL::Buffer::Data
	{
		FrameScale* frameSize;
		FrameData() = delete;
		FrameData(FrameScale* _frameSize)
			:
			Data(DynamicDraw),
			frameSize(_frameSize)
		{
		}
		virtual void* pointer()override
		{
			return nullptr;
		}
		virtual unsigned int size()override
		{
			return sizeof(Math::vec4<float>)*
				frameSize->scale.data[0] *
				frameSize->scale.data[1];
		}
	};
	struct Transform
	{

	};

	struct Model
	{
		using vec4 = Math::vec4<float>;
		using vec3 =Math::vec3<float>;
		using mat34 = Math::mat<float, 3, 4>;

		struct Color
		{
			vec4 r;
			vec4 t;
			vec3 g;
			float n;
		};
		struct TriangleData :OpenGL::Buffer::Data
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
			virtual void* pointer()
			{
				return triangles.data;
			}
			virtual unsigned int size()
			{
				return sizeof(Triangle)* triangles.length;
			}
		};
		struct PlaneData :OpenGL::Buffer::Data
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
