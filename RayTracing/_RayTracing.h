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
		struct Data
		{
			struct Perspective
			{
				double fovy;
				unsigned int y;
			};
			struct Scroll
			{
				double increaseDelta;
				double decreaseRatio;
				double threshold;
			};
			struct Key
			{
				double ratio;
			};

			Perspective persp;
			Scroll scroll;
			Key key;
			double depth;
		};
		struct Perspective
		{
			double fovy;
			unsigned int y;
			bool updated;
			Perspective()
				:
				fovy(Math::Pi* 100.0 / 180.0),
				y(1024),
				updated(false)
			{
			}
			Perspective(Data::Perspective const& _persp)
				:
				fovy(_persp.fovy),
				y(_persp.y),
				updated(false)
			{
			}
			Perspective(Perspective const&) = default;
			void init(OpenGL::FrameScale const& _size)
			{
				y = _size.h;
				updated = false;
			}
			void refresh(int _w, int _h)
			{
				y = _h;
				updated = true;
			}
		};
		struct Scroll
		{
			double increaseDelta;
			double decreaseRatio;
			double threshold;
			double total;
			Scroll()
				:
				increaseDelta(0.05),
				decreaseRatio(0.95),
				threshold(0.01),
				total(threshold)
			{
			}
			Scroll(Data::Scroll const& _scroll)
				:
				increaseDelta(_scroll.increaseDelta),
				decreaseRatio(_scroll.decreaseRatio),
				threshold(_scroll.threshold),
				total(threshold)
			{
			}
			void refresh(double _d)
			{
				total += _d * increaseDelta;
			}
			double operate()
			{
				if (abs(total) > threshold)
				{
					total *= decreaseRatio;
					return total;
				}
				else return 0.0;
			}
		};
		struct Key
		{
			bool left;
			bool right;
			bool up;
			bool down;
			double ratio;
			Key()
				:
				left(false),
				right(false),
				up(false),
				down(false),
				ratio(0.05)
			{
			}
			Key(Data::Key const& _key)
				:
				left(false),
				right(false),
				up(false),
				down(false),
				ratio(_key.ratio)
			{
			}
			void refresh(int _key, bool _operation)
			{
				switch (_key)
				{
				case 0:left = _operation; break;
				case 1:right = _operation; break;
				case 2:up = _operation; break;
				case 3:down = _operation; break;
				}
			}
			Math::vec2<double> operate()
			{
				Math::vec2<double>t
				{
					ratio * ((int)right - (int)left),
					ratio * ((int)up - (int)down)
				};
				return t;
			}
		};
		struct Mouse
		{
			struct Pointer
			{
				double x;
				double y;
				bool valid;
				Pointer()
					:
					valid(false)
				{
				}
			};
			Pointer now;
			Pointer pre;
			bool left;
			bool middle;
			bool right;
			Mouse()
				:
				now(),
				pre(),
				left(false),
				middle(false),
				right(false)
			{
			}
			void refreshPos(double _x, double _y)
			{
				if (left)
				{
					if (now.valid)
					{
						pre = now;
						now.x = _x;
						now.y = _y;
					}
					else
					{
						now.valid = true;
						now.x = _x;
						now.y = _y;
					}
				}
				else
				{
					now.valid = false;
					pre.valid = false;
				}
			}
			void refreshButton(int _button, bool _operation)
			{
				switch (_button)
				{
				case 0:	left = _operation; break;
				case 1:	middle = _operation; break;
				case 2:	right = _operation; break;
				}

			}
			Math::vec2<double> operate()
			{
				if (now.valid && pre.valid)
				{
					pre.valid = false;
					return { pre.y - now.y    ,pre.x - now.x };
				}
				else return { 0.0,0.0 };
			}
		};
		struct BufferData :OpenGL::Buffer::Data
		{
			struct Trans
			{
				Math::mat<float, 3, 4>ans;
				Math::vec3<float>r0;
				float z0;
			};
			Trans trans;
			BufferData()
				:Data(DynamicDraw)
			{
			}
			virtual void* pointer()override
			{
				return &trans;
			}
			virtual unsigned int size()override
			{
				return sizeof(Trans);
			}
		};


		Perspective persp;
		Scroll scroll;
		Key key;
		Mouse mouse;
		BufferData bufferData;
		Math::vec3<double>dr;
		Math::mat3<double>trans;
		double depth;
		bool updated;

		Transform()
			:
			persp(),
			scroll(),
			mouse(),
			updated(false),
			bufferData(),
			dr(0.0),
			trans(Math::mat3<double>::id()),
			depth(500.0)
		{
		}
		Transform(Data const& _data)
			:
			persp(_data.persp),
			scroll(_data.scroll),
			key(_data.key),
			mouse(),
			updated(false),
			bufferData(),
			dr(0.0),
			trans(Math::mat3<double>::id()),
			depth(_data.depth)
		{
		}
		void init(OpenGL::FrameScale const& _size)
		{
			persp.init(_size);
			bufferData.trans.z0 = float(_size.h) / (2.0 * tan(Math::Pi * persp.fovy / 180.0));
			calcAns();
			updated = true;
		}
		void resize(int _w, int _h)
		{
			persp.refresh(_w, _h);
		}
		void calcAns()
		{
			bufferData.trans.ans = trans;
			bufferData.trans.r0 = dr;
		}
		void operate()
		{
			Math::vec3<double>dxyz(key.operate());
			dxyz.data[2] = scroll.operate();
			Math::vec2<double>axis(mouse.operate());
			bool operated(false);
			if (dxyz != 0.0)
			{
				dr += (trans, dxyz);
				operated = true;
			}
			if (axis != 0.0)
			{
				double l(axis.length() / depth);
				trans = ((trans, Math::vec3<double>(axis)).rotMat(l), trans);
				operated = true;
			}
			if (persp.updated)
			{
				bufferData.trans.z0 = persp.y / (2.0 * tan(Math::Pi * persp.fovy / 180.0));
				persp.updated = false;
				operated = true;
			}
			if (operated)
			{
				calcAns();
				updated = true;
			}
		}
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
