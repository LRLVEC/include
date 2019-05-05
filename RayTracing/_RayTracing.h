#pragma once
#include <_Vector.h>
#include <_Math.h>
#include <_File.h>
#include <GL/_OpenGL.h>
struct STL;
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
		FrameScale()
			:
			Data(StaticDraw),
			scale{ 1024,1024 }
		{
		}
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
			Math::vec3<double> initialPosition;
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
					return { now.y - pre.y   ,now.x - pre.x };
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
		bool moved;
		bool updated;

		Transform()
			:
			persp(),
			scroll(),
			mouse(),
			moved(true),
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
			moved(true),
			updated(false),
			bufferData(),
			dr(_data.initialPosition),
			trans(Math::mat3<double>::id()),
			depth(_data.depth)
		{
		}
		void init(OpenGL::FrameScale const& _size)
		{
			persp.init(_size);
			bufferData.trans.z0 = -float(_size.h) / tan(Math::Pi * persp.fovy / 360.0);
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
			dxyz.data[2] = -scroll.operate();
			Math::vec2<double>axis(mouse.operate());
			bool operated(false);
			if (dxyz != 0.0)
			{
				dr += (trans, dxyz);
				moved = true;
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
				bufferData.trans.z0 = -(persp.y / tan(Math::Pi * persp.fovy / 360.0));
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
	struct DecayOriginData :OpenGL::Buffer::Data
	{

		DecayOriginData()
			:
			Data(DynamicDraw)
		{
		}
		virtual void* pointer()override
		{
			return nullptr;
		}
		virtual unsigned int size()override
		{
			return 16 * (8 + 1);
		}
	};
	struct Model
	{
		//Notice:	Something small can be placed in uniform buffer;
		//			but something much bigger(more than 64KB for example)
		//			must be placed in shader storage buffer...
		using vec4 = Math::vec4<float>;
		using vec3 = Math::vec3<float>;
		using vec2 = Math::vec2<float>;
		using mat34 = Math::mat<float, 3, 4>;

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
			vec3 decayFactor;
			float n;
		};

		struct Bound
		{
			struct Box
			{
				Math::mat<float, 2, 3>box;
				vec3 getCenter()const
				{
					return (box.rowVec[0] + box.rowVec[1]) / 2;
				}
				Box move(vec3 const& a)const
				{
					return
					{
						{
							box.rowVec[0] + a,
							box.rowVec[1] + a
						}
					};
				}
				Box operator+(vec3 const& a)const
				{
					return
					{
						{
							{
								box.array[0][0] <= a.data[0] ? box.array[0][0] : a.data[0],
								box.array[0][1] <= a.data[1] ? box.array[0][1] : a.data[1],
								box.array[0][2] <= a.data[2] ? box.array[0][2] : a.data[2],
							},
							{
								box.array[1][0] >= a.data[0] ? box.array[1][0] : a.data[0],
								box.array[1][1] >= a.data[1] ? box.array[1][1] : a.data[1],
								box.array[1][2] >= a.data[2] ? box.array[1][2] : a.data[2],
							}
						}
					};
				}
				Box operator+(Box const& a)const
				{
					return
					{
						{
							{
								box.array[0][0] <= a.box.array[0][0] ? box.array[0][0] : a.box.array[0][0],
								box.array[0][1] <= a.box.array[0][1] ? box.array[0][1] : a.box.array[0][1],
								box.array[0][2] <= a.box.array[0][2] ? box.array[0][2] : a.box.array[0][2],
							},
							{
								box.array[1][0] >= a.box.array[1][0] ? box.array[1][0] : a.box.array[1][0],
								box.array[1][1] >= a.box.array[1][1] ? box.array[1][1] : a.box.array[1][1],
								box.array[1][2] >= a.box.array[1][2] ? box.array[1][2] : a.box.array[1][2]
							}
						}
					};
				}
				Box& operator+=(vec3 const& a)
				{
					box.array[0][0] = box.array[0][0] <= a.data[0] ? box.array[0][0] : a.data[0];
					box.array[0][1] = box.array[0][1] <= a.data[1] ? box.array[0][1] : a.data[1];
					box.array[0][2] = box.array[0][2] <= a.data[2] ? box.array[0][2] : a.data[2];
					box.array[1][0] = box.array[1][0] >= a.data[0] ? box.array[1][0] : a.data[0];
					box.array[1][1] = box.array[1][1] >= a.data[1] ? box.array[1][1] : a.data[1];
					box.array[1][2] = box.array[1][2] >= a.data[2] ? box.array[1][2] : a.data[2];
					return *this;
				}
				Box& operator+=(Box const& a)
				{
					box.array[0][0] = box.array[0][0] <= a.box.array[0][0] ? box.array[0][0] : a.box.array[0][0];
					box.array[0][1] = box.array[0][1] <= a.box.array[0][1] ? box.array[0][1] : a.box.array[0][1];
					box.array[0][2] = box.array[0][2] <= a.box.array[0][2] ? box.array[0][2] : a.box.array[0][2];
					box.array[1][0] = box.array[1][0] >= a.box.array[1][0] ? box.array[1][0] : a.box.array[1][0];
					box.array[1][1] = box.array[1][1] >= a.box.array[1][1] ? box.array[1][1] : a.box.array[1][1];
					box.array[1][2] = box.array[1][2] >= a.box.array[1][2] ? box.array[1][2] : a.box.array[1][2];
					return *this;
				}
				bool operator==(Box const& a)const
				{
					return box == a.box;
				}
			};

			Box box;
			vec3 center;
			float area;

			Bound move(vec3 const& a)const
			{
				return
				{
					box.move(a),
					center + a ,
					area
				};
			}
			Bound operator+(vec3 const& a)const
			{
				Box t(box + a);
				return{ t,t.getCenter(),area };
			}
			Bound operator+(Bound const& a)const
			{
				Box t(box + a.box);
				return { t,t.getCenter(),area + a.area };
			}
			Bound& operator+=(vec3 const& a)
			{
				box += a;
				center = box.getCenter();
				return *this;
			}
			Bound& operator+=(Bound const& a)
			{
				box += a.box;
				center = box.getCenter();
				area += a.area;
				return *this;
			}

		};


		struct Planes
		{
			struct PlaneData :OpenGL::Buffer::Data
			{
				struct Plane
				{
					vec4 paras;	//Ax + By + Cz + W = 0, this is (A, B, C, W).
					Color color;
				};
				Vector<Plane>planes;
				PlaneData()
					:
					Data(DynamicDraw)
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
			struct Info
			{
				OpenGL::BufferType type;
				int index;
			};

			PlaneData data;
			OpenGL::Buffer buffer;
			OpenGL::BufferConfig config;
			bool numChanged;
			bool upToDate;
			Planes(Info const& _info)
				:
				buffer(&data),
				config(&buffer, _info.type, _info.index),
				numChanged(false),
				upToDate(true)
			{
			}
			void dataInit()
			{
				if (numChanged)
				{
					config.dataInit();
				}
				else if (!upToDate)
				{
					config.refreshData();
				}
				upToDate = true;
			}
		};
		struct Triangles
		{
			struct TriangleOriginData :OpenGL::Buffer::Data
			{
				struct TriangleOrigin
				{
					mat34 vertices;
					vec2 uv1;
					vec2 uv2;
					vec4 uv3;
					Math::vec4<int>nIndices;
					Color color;

					Bound bound()const
					{
						Bound::Box a
						{
							{
								{
									vertices.column(0).min(),
									vertices.column(1).min(),
									vertices.column(2).min()
								},
								{
									vertices.column(0).max(),
									vertices.column(1).max(),
									vertices.column(2).max()
								},
							}
						};
						return
						{
							a,
							a.getCenter(),
							(
								vec3(vertices.rowVec[1] - vertices.rowVec[0]) |
								vec3(vertices.rowVec[2] - vertices.rowVec[0])
							).length() / 2
						};
					}
				};
				Vector<TriangleOrigin>trianglesOrigin;
				TriangleOriginData()
					:
					Data(DynamicDraw)
				{
				}
				virtual void* pointer()
				{
					return trianglesOrigin.data;
				}
				virtual unsigned int size()
				{
					return sizeof(TriangleOrigin)* trianglesOrigin.length;
				}
			};
			struct TriangleGPUData :OpenGL::Buffer::Data
			{
				struct TriangleGPU
				{
					vec4 plane;
					vec4 p1;
					vec4 k1;
					vec4 k2;
					vec2 uv1;
					vec2 uv2;
					vec4 uv3;
					Math::vec4<int>nIndices;
					Color color;
				};
				unsigned int num;
				TriangleGPUData()
					:
					OpenGL::Buffer::Data(DynamicDraw),
					num(0)
				{
				}
				virtual void* pointer()
				{
					return nullptr;
				}
				virtual unsigned int size()
				{
					return sizeof(TriangleGPU)* num;
				}
			};
			struct Info
			{
				int indexOrigin;
				int indexGPU;
			};
			TriangleOriginData trianglesOrigin;
			TriangleGPUData trianglesGPU;
			OpenGL::Buffer trianglesOriginBuffer;
			OpenGL::Buffer trianglesGPUBuffer;
			OpenGL::BufferConfig trianglesOriginConfig;
			OpenGL::BufferConfig trianglesGPUConfig;
			bool numChanged;
			bool originUpToDate;
			bool GPUUpToDate;
			Triangles(Info const& _info)
				:
				trianglesOriginBuffer(&trianglesOrigin),
				trianglesGPUBuffer(&trianglesGPU),
				trianglesOriginConfig(&trianglesOriginBuffer, OpenGL::ShaderStorageBuffer, _info.indexOrigin),
				trianglesGPUConfig(&trianglesGPUBuffer, OpenGL::ShaderStorageBuffer, _info.indexGPU),
				numChanged(true),
				originUpToDate(false),
				GPUUpToDate(false)
			{
			}
			void dataInit()
			{
				if (numChanged)
				{
					trianglesOriginConfig.dataInit();
					trianglesGPU.num = trianglesOrigin.trianglesOrigin.length;
					trianglesGPUConfig.dataInit();
					GPUUpToDate = false;
				}
				else if (!originUpToDate)
				{
					trianglesOriginConfig.refreshData();
					GPUUpToDate = false;
				}
				if (!trianglesOrigin.trianglesOrigin.length)
					GPUUpToDate = false;
				originUpToDate = true;
			}
		};
		struct Spheres
		{
			struct SphereData :OpenGL::Buffer::Data
			{
				struct Sphere
				{
					vec4 sphere;
					vec4 e1;
					vec4 e2;
					Color color;

					Bound bound()const
					{
						float r = 1.05 * sqrt(sphere.data[3]);
						return
						{
							{
								{
									vec3(sphere) - vec3(r),
									vec3(sphere) + vec3(r)
								}
							},
							sphere,
							float(4 * Math::Pi * sphere.data[3])
						};
					}
				};
				Vector<Sphere>spheres;
				SphereData()
					:
					Data(DynamicDraw)
				{
				}
				virtual void* pointer()override
				{
					return spheres.data;
				}
				virtual unsigned int size()override
				{
					return sizeof(Sphere)* spheres.length;
				}
			};
			struct Info
			{
				int index;
			};

			SphereData data;
			OpenGL::Buffer buffer;
			OpenGL::BufferConfig config;
			bool numChanged;
			bool upToDate;
			bool GPUUpToDate;
			Spheres(Info const& _info)
				:
				buffer(&data),
				config(&buffer, OpenGL::ShaderStorageBuffer, _info.index),
				numChanged(true),
				upToDate(false),
				GPUUpToDate(false)
			{
			}
			void dataInit()
			{
				if (numChanged)
				{
					config.dataInit();
				}
				else if (!upToDate)
				{
					config.refreshData();
					GPUUpToDate = false;
				}
				upToDate = true;
			}
		};
		struct Circles
		{
			struct CircleData :OpenGL::Buffer::Data
			{
				struct Circle
				{
					vec4 plane;		//n(unnormalized)
					vec3 sphere;	//p, R^2
					float r2;
					vec4 e1;			//e(unnormalized)
					Color color;
					void init()
					{
						plane.normaliaze(3);
						e1.normaliaze(3);
					}
					Bound bound()const
					{
						vec3 dr
						{
							sqrtf(1 - pow((vec3(plane), vec3({ 1,0,0 })), 2)) ,
							sqrtf(1 - pow((vec3(plane), vec3({ 0,1,0 })), 2)) ,
							sqrtf(1 - pow((vec3(plane), vec3({ 0,0,1 })), 2))
						};
						dr *= 1.05 * sqrt(r2);
						return
						{
							{
								{
									sphere - dr,
									sphere + dr
								}
							},
							sphere,
							float(Math::Pi * r2)
						};
					}
				};
				Vector<Circle>circles;
				CircleData()
					:
					Data(DynamicDraw)
				{
				}
				virtual void* pointer()override
				{
					return circles.data;
				}
				virtual unsigned int size()override
				{
					return sizeof(Circle)* circles.length;
				}
			};
			struct Info
			{
				int index;
			};
			CircleData data;
			OpenGL::Buffer buffer;
			OpenGL::BufferConfig config;
			bool numChanged;
			bool upToDate;
			bool GPUUpToDate;
			Circles(Info const& _info)
				:
				buffer(&data),
				config(&buffer, OpenGL::ShaderStorageBuffer, _info.index),
				numChanged(true),
				upToDate(false),
				GPUUpToDate(false)
			{
			}
			void dataInit()
			{
				if (numChanged)
				{
					config.dataInit();
					GPUUpToDate = false;
				}
				else if (!upToDate)
				{
					config.refreshData();
					GPUUpToDate = false;
				}
				upToDate = true;
			}
		};
		struct Cylinders
		{
			struct CylinderData :OpenGL::Buffer::Data
			{
				struct Cylinder
				{
					vec3 c;
					float r2;
					vec3 n;
					float l;
					vec4 e1;
					Color color;
					void init()
					{
						n.normaliaze(3);
						e1.normaliaze(3);
					}
					Bound bound()const
					{
						vec3 dr
						{
							sqrtf(1 - pow((n, vec3({ 1,0,0 })), 2)) ,
							sqrtf(1 - pow((n, vec3({ 0,1,0 })), 2)) ,
							sqrtf(1 - pow((n, vec3({ 0,0,1 })), 2))
						};
						dr *= 1.05 * sqrt(r2);
						Bound::Box a{ {c - dr, c + dr} };
						return
						{
							a += a.move(l * n),
							c + (l / 2) * n,
							float(2 * Math::Pi * l * sqrt(r2))
						};
					}
				};
				Vector<Cylinder> cylinders;
				CylinderData()
					:
					Data(DynamicDraw)
				{
				}
				virtual void* pointer()override
				{
					return cylinders.data;
				}
				virtual unsigned int size()override
				{
					return sizeof(Cylinder)* cylinders.length;
				}
			};
			struct Info
			{
				int index;
			};
			CylinderData data;
			OpenGL::Buffer buffer;
			OpenGL::BufferConfig config;
			bool numChanged;
			bool upToDate;
			bool GPUUpToDate;
			Cylinders(Info const& _info)
				:
				buffer(&data),
				config(&buffer, OpenGL::ShaderStorageBuffer, _info.index),
				numChanged(true),
				upToDate(false),
				GPUUpToDate(false)
			{
			}
			void dataInit()
			{
				if (numChanged)
				{
					config.dataInit();
					GPUUpToDate = false;
				}
				else if (!upToDate)
				{
					config.refreshData();
					GPUUpToDate = false;
				}
				upToDate = true;
			}
		};
		struct Cones
		{
			struct ConeData :OpenGL::Buffer::Data
			{
				struct Cone
				{
					vec3 c;
					float c2;
					vec3 n;
					float l2;
					vec4 e1;
					Color color;
					void init()
					{
						n.normaliaze();
						e1.normaliaze(3);
					}
					Bound bound()const
					{
						vec3 dr
						{
							sqrtf(1 - pow(n.data[0],2)) ,
							sqrtf(1 - pow(n.data[1],2)) ,
							sqrtf(1 - pow(n.data[2],2))
						};
						dr *= 1.05 * sqrt(l2 * (1 - c2));
						vec3 cc(c + n * sqrt(l2 * c2));
						Bound::Box a{ {cc - dr, cc + dr} };
						return
						{
							a.operator+(c),
							0.75 * sqrt(l2 * c2) * n + c,
							float(Math::Pi * l2 * sqrt(1 - c2))
						};
					}
				};
				Vector<Cone>cones;
				ConeData()
					:
					Data(DynamicDraw)
				{
				}
				virtual void* pointer()override
				{
					return cones.data;
				}
				virtual unsigned int size()override
				{
					return sizeof(Cone)* cones.length;
				}
			};
			struct Info
			{
				int index;
			};
			ConeData data;
			OpenGL::Buffer buffer;
			OpenGL::BufferConfig config;
			bool numChanged;
			bool upToDate;
			bool GPUUpToDate;
			Cones(Info const& _info)
				:
				buffer(&data),
				config(&buffer, OpenGL::ShaderStorageBuffer, _info.index),
				numChanged(true),
				upToDate(false),
				GPUUpToDate(false)
			{
			}
			void dataInit()
			{
				if (numChanged)
				{
					config.dataInit();
					GPUUpToDate = false;
				}
				else if (!upToDate)
				{
					config.refreshData();
					GPUUpToDate = false;
				}
				upToDate = true;
			}
		};
		struct PointLights
		{
			struct PointLightData :OpenGL::Buffer::Data
			{
				struct PointLight
				{
					vec4 color;
					vec4 p;
				};
				Vector<PointLight>pointLights;
				PointLightData()
					:
					Data(StaticDraw)
				{
				}
				virtual void* pointer()override
				{
					return pointLights.data;
				}
				virtual unsigned int size()override
				{
					return sizeof(PointLight)* pointLights.length;
				}
			};
			struct Info
			{
				int index;
			};
			PointLightData data;
			OpenGL::Buffer buffer;
			OpenGL::BufferConfig config;
			bool numChanged;
			bool upToDate;
			bool GPUUpToDate;
			PointLights(Info const& _info)
				:
				buffer(&data),
				config(&buffer, OpenGL::ShaderStorageBuffer, _info.index),
				numChanged(true),
				upToDate(false),
				GPUUpToDate(false)
			{
			}
			void dataInit()
			{
				if (numChanged)
				{
					config.dataInit();
					GPUUpToDate = false;
				}
				else if (!upToDate)
				{
					config.refreshData();
					GPUUpToDate = false;
				}
				upToDate = true;
			}
		};
		struct GeometryNum
		{
			struct NumData :OpenGL::Buffer::Data
			{
				struct Num
				{
					unsigned int planeNum;
					unsigned int triangleNum;
					unsigned int sphereNum;
					unsigned int circleNum;
					unsigned int cylinderNum;
					unsigned int coneNum;
					unsigned int pointLightNum;
					unsigned int blank[1];//补齐考虑一下
					Num()
						:
						planeNum(0),
						triangleNum(0),
						sphereNum(0),
						circleNum(0),
						cylinderNum(0),
						pointLightNum(0)
					{
					}
				};
				Num num;
				NumData()
					:
					Data(StaticDraw),
					num()
				{
				}
				NumData(Num const& _num)
					:
					Data(StaticDraw),
					num(_num)
				{
				}
				virtual void* pointer()override
				{
					return &num;
				}
				virtual unsigned int size()override
				{
					return sizeof(Num);
				}
			};
			struct Info
			{
				int index;
			};
			NumData data;
			OpenGL::Buffer buffer;
			OpenGL::BufferConfig config;
			GeometryNum(Info const& _info)
				:
				buffer(&data),
				config(&buffer, OpenGL::UniformBuffer, _info.index)
			{
			}
			void dataInit()
			{
				config.dataInit();
			}
		};
		struct Header
		{
			GeometryNum::NumData::Num num;
			GeometryNum::NumData::Num offset;
		};
		struct BVH
		{
			struct Node
			{
				struct NodeGPU
				{
					vec3 min;
					unsigned int leftChild;
					vec3 max;
					unsigned int rightChild;
					unsigned int fatherIndex;
					unsigned int axis;
					unsigned int geometry;
					unsigned int geometryNum;
					unsigned int blank[4];
					NodeGPU() = default;
					NodeGPU(Node const& node, unsigned int father)
						:
						min(node.boundAll.box.box.rowVec[0]),
						leftChild(0),
						max(node.boundAll.box.box.rowVec[1]),
						rightChild(0),
						fatherIndex(father),
						axis(node.axis),
						geometry(node.geometry),
						geometryNum(node.geometryNum)
					{
					}
				};
				Node* childs[2];
				Bound boundAll;
				unsigned int axis;
				int geometry;
				unsigned int geometryNum;

				Node()
					:
					axis(0),
					geometry(0),
					geometryNum(0)
				{
					childs[0] = childs[1] = nullptr;
				}
				Node(Bound _bound, unsigned int _geometry, unsigned int _geometryNum)
					:
					boundAll(_bound),
					geometry(_geometry),
					geometryNum(_geometryNum),
					axis(0)
				{
					childs[0] = childs[1] = nullptr;
				}
				Node(Node* nodes, Vector<unsigned int>const& indices)
					:
					geometry(0),
					geometryNum(0)
				{
					childs[0] = childs[1] = nullptr;
					boundAll = nodes[indices.data[0]].boundAll;
					vec3 c(nodes[indices.data[0]].boundAll.area * nodes[indices.data[0]].boundAll.center);
					for (int c0(1); c0 < indices.length; ++c0)
					{
						boundAll += nodes[indices.data[c0]].boundAll;
						c += nodes[indices.data[c0]].boundAll.area * nodes[indices.data[c0]].boundAll.center;
					}
					c /= boundAll.area;
					vec3 variance(0);
					for (int c0(0); c0 < indices.length; ++c0)
					{
						vec3 temp(nodes[indices.data[c0]].boundAll.center - c);
						variance += nodes[indices.data[c0]].boundAll.area * (temp * temp);
					}
					axis =
						variance[1] > variance[0] ?
						variance[1] > variance[2] ? 1 : 2 :
						variance[0] > variance[2] ? 0 : 2;
					Vector<unsigned int>indicesLeft;
					Vector<unsigned int>indicesRight;
					for (int c0(0); c0 < indices.length; ++c0)
					{
						if (nodes[indices.data[c0]].boundAll.box == boundAll.box && !geometry)
						{
							geometry = nodes[indices.data[c0]].geometry;
							geometryNum = nodes[indices.data[c0]].geometryNum;
							continue;
						}
						if (nodes[indices.data[c0]].boundAll.center[axis] < c[axis])
							indicesLeft.pushBack(indices.data[c0]);
						else
							indicesRight.pushBack(indices.data[c0]);
					}
					if (indicesLeft.length > 1)
					{
						childs[0] = (Node*)::malloc(sizeof(Node));
						new(childs[0])Node(nodes, indicesLeft);
						if (indicesRight.length > 1)
						{
							childs[1] = (Node*)::malloc(sizeof(Node));
							new(childs[1])Node(nodes, indicesRight);
						}
						else if(indicesRight.length)
							childs[1] = nodes + indicesRight[0];
					}
					else
					{
						if (indicesLeft.length)
						{
							childs[0] = nodes + indicesLeft[0];
							if (indicesRight.length > 1)
							{
								childs[1] = (Node*)::malloc(sizeof(Node));
								new(childs[1])Node(nodes, indicesRight);
							}
							else if (indicesRight.length)
								childs[1] = nodes + indicesRight[0];
						}
						else
						{
							if (indicesRight.length > 1)
							{
								childs[0] = (Node*)::malloc(sizeof(Node));
								new(childs[0])Node(nodes, indicesRight);
							}
							else
								childs[0] = nodes + indicesRight[0];
						}
					}
				}
				~Node()
				{
					if (childs[0] && childs[0]->childs[0])
						free(childs[0]);
					if (childs[1] && childs[1]->childs[0])
						free(childs[1]);
				}
				static void getLinearBVH(Vector<NodeGPU>& nodeGPU, Node const& father, unsigned int fatherIndex)
				{
					nodeGPU[fatherIndex].leftChild = 1;
					nodeGPU.pushBack({ *father.childs[0] ,fatherIndex });
					if (father.childs[0]->childs[0])
						getLinearBVH(nodeGPU, *father.childs[0], nodeGPU.length - 1);
					if (father.childs[1])
					{
						nodeGPU.pushBack({ *father.childs[1] ,fatherIndex });
						nodeGPU[fatherIndex].rightChild = nodeGPU.length - 1;
						if (father.childs[1]->childs[0])
							getLinearBVH(nodeGPU, *father.childs[1], nodeGPU.length - 1);
					}
				}
			};
			struct BVHData :OpenGL::Buffer::Data
			{
				Vector<Node::NodeGPU> linearBVH;
				BVHData()
					:
					Data(DynamicDraw)
				{
				}
				virtual void* pointer()override
				{
					return linearBVH.data;
				}
				virtual unsigned int size()override
				{
					return sizeof(Node::NodeGPU)* linearBVH.length;
				}
			};
			struct Info
			{
				int index;
			};

			Model* model;
			Vector<Node>nodes;
			Node father;
			BVHData data;
			OpenGL::Buffer buffer;
			OpenGL::BufferConfig config;
			BVH(Info const& _info, Model* _model)
				:
				model(_model),
				buffer(&data),
				config(&buffer, OpenGL::ShaderStorageBuffer, _info.index)
			{
			}
			void getBounds()
			{
				Vector<Triangles::TriangleOriginData::TriangleOrigin>const& triangles(model->triangles.trianglesOrigin.trianglesOrigin);
				Vector<Spheres::SphereData::Sphere>const& spheres(model->spheres.data.spheres);
				Vector<Circles::CircleData::Circle>const& circles(model->circles.data.circles);
				Vector<Cylinders::CylinderData::Cylinder>const& cylinders(model->cylinders.data.cylinders);
				Vector<Cones::ConeData::Cone>const& cones(model->cones.data.cones);
				for (unsigned int c0(0); c0 < triangles.length; ++c0)
					nodes.pushBack({ triangles.data[c0].bound(),2,c0 });
				for (unsigned int c0(0); c0 < spheres.length; ++c0)
					nodes.pushBack({ spheres.data[c0].bound(), 3, c0 });
				for (unsigned int c0(0); c0 < circles.length; ++c0)
					nodes.pushBack({ circles.data[c0].bound(), 4, c0 });
				for (unsigned int c0(0); c0 < cylinders.length; ++c0)
					nodes.pushBack({ cylinders.data[c0].bound(), 5, c0 });
				for (unsigned int c0(0); c0 < cones.length; ++c0)
					nodes.pushBack({ cones.data[c0].bound(), 6, c0 });
			}
			//Still some special cases not considered...
			//(for example, all centers are the same, and there is no bounding relationship...)
			//(just create a list to solve this...)
			void getBVH()
			{
				vec3 variance(0);
				nodes.~Vector();
				data.linearBVH.~Vector();
				nodes.length = nodes.lengthAll = 0;
				data.linearBVH.length = data.linearBVH.lengthAll = 0;
				getBounds();
				father.boundAll = nodes[0].boundAll;
				if (nodes.length > 1)
				{
					vec3 c = nodes[0].boundAll.area * nodes[0].boundAll.center;
					for (int c0(1); c0 < nodes.length; ++c0)
					{
						father.boundAll += nodes[c0].boundAll;
						c += nodes[c0].boundAll.area * nodes[c0].boundAll.center;
					}
					c /= father.boundAll.area;
					for (int c0(0); c0 < nodes.length; ++c0)
					{
						vec3 temp(nodes[c0].boundAll.center - c);
						variance += nodes[c0].boundAll.area * (temp * temp);
					}
					unsigned int axis;
					axis =
						variance[1] > variance[0] ?
						variance[1] > variance[2] ? 1 : 2 :
						variance[0] > variance[2] ? 0 : 2;
					father.axis = axis;
					Vector<unsigned int>indicesLeft;
					Vector<unsigned int>indicesRight;
					for (int c0(0); c0 < nodes.length; ++c0)
					{
						if (nodes[c0].boundAll.box == father.boundAll.box && !father.geometry)
						{
							father.geometry = nodes[c0].geometry;
							father.geometryNum = nodes[c0].geometryNum;
							continue;
						}
						if (nodes[c0].boundAll.center[axis] < c[axis])
							indicesLeft.pushBack(c0);
						else
							indicesRight.pushBack(c0);
					}
					if (indicesLeft.length > 1)
					{
						father.childs[0] = (Node*)::malloc(sizeof(Node));
						new(father.childs[0])Node(nodes.data, indicesLeft);
						if (indicesRight.length > 1)
						{
							father.childs[1] = (Node*)::malloc(sizeof(Node));
							new(father.childs[1])Node(nodes.data, indicesRight);
						}
						else if (indicesRight.length)
							father.childs[1] = nodes.data + indicesRight[0];
					}
					else if (indicesLeft.length)
					{
						father.childs[0] = nodes.data + indicesLeft[0];
						if (indicesRight.length > 1)
						{
							father.childs[1] = (Node*)::malloc(sizeof(Node));
							new(father.childs[1])Node(nodes.data, indicesRight);
						}
						else if (indicesRight.length)
							father.childs[1] = nodes.data + indicesRight[0];
					}
					else
					{
						if (indicesRight.length > 1)
						{
							father.childs[0] = (Node*)::malloc(sizeof(Node));
							new(father.childs[0])Node(nodes.data, indicesRight);
						}
						else
							father.childs[0] = nodes.data + indicesRight[0];
					}

				}
				else
				{
					father.geometry = nodes[0].geometry;
					father.geometryNum = nodes[0].geometryNum;
					father.boundAll = nodes[0].boundAll;
				}
				if (!nodes.length)return;
				data.linearBVH.pushBack({ father,0 });
				if (father.childs[0])
					Node::getLinearBVH(data.linearBVH, father, 0);
			}
			void dataInit()
			{
				// timer;
				//timer.begin();
				getBVH();
				config.dataInit();
				//timer.end();
				//timer.print("build time:");
			}
		};
		struct Info
		{
			Planes::Info planesInfo;
			Triangles::Info trianglesInfo;
			Spheres::Info spheresInfo;
			Circles::Info circlesInfo;
			Cylinders::Info cylindersInfo;
			Cones::Info conesInfo;
			PointLights::Info pointLightsInfo;
			GeometryNum::Info geometryNumInfo;
			BVH::Info bvhInfo;
		};

		Planes planes;
		Triangles triangles;
		Spheres spheres;
		Circles circles;
		Cylinders cylinders;
		Cones cones;
		PointLights pointLights;
		GeometryNum geometryNum;
		BVH bvh;
		bool moved;

		Model()
			:
			planes({ OpenGL::None,-1 }),
			triangles({ -1,-1 }),
			spheres({ -1 }),
			circles({ -1 }),
			cylinders({ -1 }),
			cones({ -1 }),
			pointLights({ -1 }),
			geometryNum({ -1 }),
			bvh({ -1 }, this),
			moved(true)
		{
		}
		Model(Info const& _info)
			:
			planes(_info.planesInfo),
			triangles(_info.trianglesInfo),
			spheres(_info.spheresInfo),
			circles(_info.circlesInfo),
			cylinders(_info.cylindersInfo),
			cones(_info.conesInfo),
			pointLights(_info.pointLightsInfo),
			geometryNum(_info.geometryNumInfo),
			bvh(_info.bvhInfo, this),
			moved(true)
		{
		}
		void dataInit()
		{
			bool numChanged(false);
			planes.dataInit();
			triangles.dataInit();
			spheres.dataInit();
			circles.dataInit();
			cylinders.dataInit();
			cones.dataInit();
			pointLights.dataInit();
			if (planes.numChanged)
			{
				geometryNum.data.num.planeNum = planes.data.planes.length;
				planes.numChanged = false;
				numChanged = true;
			}
			if (triangles.numChanged)
			{
				geometryNum.data.num.triangleNum = triangles.trianglesOrigin.trianglesOrigin.length;
				triangles.numChanged = false;
				numChanged = true;
			}
			if (spheres.numChanged)
			{
				geometryNum.data.num.sphereNum = spheres.data.spheres.length;
				spheres.numChanged = false;
				numChanged = true;
			}
			if (circles.numChanged)
			{
				geometryNum.data.num.circleNum = circles.data.circles.length;
				circles.numChanged = false;
				numChanged = true;
			}
			if (cylinders.numChanged)
			{
				geometryNum.data.num.cylinderNum = cylinders.data.cylinders.length;
				cylinders.numChanged = false;
				numChanged = true;
			}
			if (cones.numChanged)
			{
				geometryNum.data.num.coneNum = cones.data.cones.length;
				cones.numChanged = false;
				numChanged = true;
			}
			if (pointLights.numChanged)
			{
				geometryNum.data.num.pointLightNum = pointLights.data.pointLights.length;
				pointLights.numChanged = false;
				numChanged = true;
			}
			if (numChanged)
			{
				geometryNum.dataInit();
			}
			moved |=
				numChanged |
				(!triangles.GPUUpToDate) |
				(!spheres.GPUUpToDate) |
				(!circles.GPUUpToDate) |
				(!cylinders.GPUUpToDate) |
				(!cones.GPUUpToDate);
			if (moved)
				bvh.dataInit();
		}
		void upToDate()
		{
			triangles.GPUUpToDate = true;
			spheres.GPUUpToDate = true;
			circles.GPUUpToDate = true;
			cylinders.GPUUpToDate = true;
			cones.GPUUpToDate = true;
			moved = false;
		}
		void addSTL(STL const&, Color const&,unsigned int);
		void addCylinder(Cylinders::CylinderData::Cylinder const& _cylinder)
		{
			Circles::CircleData::Circle circle0;
			Circles::CircleData::Circle circle1;
			cylinders.data.cylinders.pushBack(_cylinder);
			circles.data.circles.pushBack
			(
				{
					-_cylinder.n,
					_cylinder.c,
					_cylinder.r2,
					_cylinder.e1,
					_cylinder.color
				}
			);
			circles.data.circles.pushBack
			(
				{
					_cylinder.n,
					_cylinder.c + _cylinder.n * _cylinder.l,
					_cylinder.r2,
					_cylinder.e1,
					_cylinder.color
				}
			);
			circles.numChanged = true;
			cylinders.numChanged = true;
		}
		void addCone(Cones::ConeData::Cone const& _cone)
		{
			circles.data.circles.pushBack
			(
				{
					_cone.n,
					_cone.c + _cone.n * sqrtf(_cone.l2 * _cone.c2),
					_cone.l2 * (1 - _cone.c2),
					_cone.e1,
					_cone.color
				}
			);
			cones.data.cones.pushBack(_cone);
			circles.numChanged = true;
			cones.numChanged = true;
		}
		void readModel(String<char>const& _path)
		{
			FILE* temp(::fopen(_path.data, "rb+"));
			::fseek(temp, 0, SEEK_SET);
			Model::Header header;
			::fread(&header, 1, sizeof(Model::Header), temp);
			geometryNum.data.num = header.num;
			if (header.num.planeNum)
			{
				::fseek(temp, header.offset.planeNum, SEEK_SET);
				planes.data.planes.malloc(header.num.planeNum);
				::fread(planes.data.planes.data, header.num.planeNum, sizeof(Model::Planes::PlaneData::Plane), temp);
			}
			if (header.num.triangleNum)
			{
				::fseek(temp, header.offset.triangleNum, SEEK_SET);
				triangles.trianglesOrigin.trianglesOrigin.malloc(header.num.triangleNum);
				::fread((triangles.trianglesOrigin.trianglesOrigin.data), header.num.triangleNum, sizeof(Model::Triangles::TriangleOriginData::Data), temp);
			}
			if (header.num.sphereNum)
			{
				::fseek(temp, header.offset.sphereNum, SEEK_SET);
				spheres.data.spheres.malloc(header.num.sphereNum);
				::fread((spheres.data.spheres.data), header.num.sphereNum, sizeof(Model::Spheres::SphereData::Data), temp);
			}
			if (header.num.circleNum)
			{
				::fseek(temp, header.offset.circleNum, SEEK_SET);
				circles.data.circles.malloc(header.num.circleNum);
				::fread((circles.data.circles.data), header.num.circleNum, sizeof(Model::Circles::CircleData::Data), temp);
			}
			if (header.num.cylinderNum)
			{
				::fseek(temp, header.offset.cylinderNum, SEEK_SET);
				cylinders.data.cylinders.malloc(header.num.cylinderNum);
				::fread((cylinders.data.cylinders.data), header.num.cylinderNum, sizeof(Model::Cylinders::CylinderData::Data), temp);
			}
			if (header.num.coneNum)
			{
				::fseek(temp, header.offset.coneNum, SEEK_SET);
				cones.data.cones.malloc(header.num.coneNum);
				::fread((cones.data.cones.data), header.num.coneNum, sizeof(Model::Cones::ConeData::Data), temp);
			}
			if (header.num.pointLightNum)
			{
				::fseek(temp, header.offset.pointLightNum, SEEK_SET);
				pointLights.data.pointLights.malloc(header.num.pointLightNum);
				::fread((pointLights.data.pointLights.data), header.num.pointLightNum, sizeof(Model::PointLights::PointLightData::Data), temp);
			}
			::fclose(temp);
		}
		void createModel(String<char>const& _path)
		{
			FILE* temp(::fopen(_path.data, "wb+"));
			::fseek(temp, 0, SEEK_SET);
			Header header;
			header.num = geometryNum.data.num;
			header.offset.planeNum = sizeof(Header);
			header.offset.triangleNum = header.num.planeNum * sizeof(Model::Planes::PlaneData::Plane) + header.offset.planeNum;
			header.offset.sphereNum = header.num.triangleNum * sizeof(Model::Triangles::TriangleOriginData::Data) + header.offset.triangleNum;
			header.offset.circleNum = header.num.sphereNum * sizeof(Model::Spheres::SphereData::Data) + header.offset.sphereNum;
			header.offset.cylinderNum = header.num.circleNum * sizeof(Model::Circles::CircleData::Data) + header.offset.circleNum;
			header.offset.coneNum = header.num.cylinderNum * sizeof(Model::Cylinders::CylinderData::Data) + header.offset.cylinderNum;
			header.offset.pointLightNum = header.num.coneNum * sizeof(Model::Cones::ConeData::Data) + header.offset.coneNum;
			::fwrite(&header, 1, sizeof(header), temp);//把头写入；
			if (header.num.planeNum)
			{
				::fseek(temp, header.offset.planeNum, SEEK_SET);
				::fwrite(planes.data.planes.data, header.num.planeNum, sizeof(Model::Planes::PlaneData::Plane), temp);
			}
			if (header.num.triangleNum)
			{
				::fseek(temp, header.offset.triangleNum, SEEK_SET);
				::fwrite((triangles.trianglesOrigin.trianglesOrigin.data), header.num.triangleNum, sizeof(Model::Triangles::TriangleOriginData::Data), temp);
			}
			if (header.num.sphereNum)
			{
				::fseek(temp, header.offset.sphereNum, SEEK_SET);
				::fwrite((spheres.data.spheres.data), header.num.sphereNum, sizeof(Model::Spheres::SphereData::Data), temp);
			}
			if (header.num.circleNum)
			{
				::fseek(temp, header.offset.circleNum, SEEK_SET);
				::fwrite((circles.data.circles.data), header.num.circleNum, sizeof(Model::Circles::CircleData::Data), temp);
			}
			if (header.num.cylinderNum)
			{
				::fseek(temp, header.offset.cylinderNum, SEEK_SET);
				::fwrite((cylinders.data.cylinders.data), header.num.cylinderNum, sizeof(Model::Cylinders::CylinderData::Data), temp);
			}
			if (header.num.coneNum)
			{
				::fseek(temp, header.offset.coneNum, SEEK_SET);
				::fwrite((cones.data.cones.data), header.num.coneNum, sizeof(Model::Cones::ConeData::Data), temp);
			}
			if (header.num.pointLightNum)
			{
				::fseek(temp, header.offset.pointLightNum, SEEK_SET);
				::fwrite((pointLights.data.pointLights.data), header.num.pointLightNum, sizeof(Model::PointLights::PointLightData::Data), temp);
			}
			::fclose(temp);
		}

	};
}
inline RayTracing::Model File::readModel()const
{
	if (!this)
		return RayTracing::Model();
	RayTracing::Model r;
	FILE* temp(::fopen((property.path + property.file.name).data, "rb+"));
	::fseek(temp, 0, SEEK_SET);
	RayTracing::Model::Header header;
	::fread(&header, 1, sizeof(RayTracing::Model::Header), temp);
	r.geometryNum.data.num = header.num;
	if (header.num.planeNum)
	{
		::fseek(temp, header.offset.planeNum, SEEK_SET);
		r.planes.data.planes.malloc(header.num.planeNum);
		::fread(r.planes.data.planes.data, header.num.planeNum, sizeof(RayTracing::Model::Planes::PlaneData::Plane), temp);
	}
	if (header.num.triangleNum)
	{
		::fseek(temp, header.offset.triangleNum, SEEK_SET);
		r.triangles.trianglesOrigin.trianglesOrigin.malloc(header.num.triangleNum);
		::fread((r.triangles.trianglesOrigin.trianglesOrigin.data), header.num.triangleNum, sizeof(RayTracing::Model::Triangles::TriangleOriginData::Data), temp);
	}
	if (header.num.sphereNum)
	{
		::fseek(temp, header.offset.sphereNum, SEEK_SET);
		r.spheres.data.spheres.malloc(header.num.sphereNum);
		::fread((r.spheres.data.spheres.data), header.num.sphereNum, sizeof(RayTracing::Model::Spheres::SphereData::Data), temp);
	}
	if (header.num.circleNum)
	{
		::fseek(temp, header.offset.circleNum, SEEK_SET);
		r.circles.data.circles.malloc(header.num.circleNum);
		::fread((r.circles.data.circles.data), header.num.circleNum, sizeof(RayTracing::Model::Circles::CircleData::Data), temp);
	}
	if (header.num.cylinderNum)
	{
		::fseek(temp, header.offset.cylinderNum, SEEK_SET);
		r.cylinders.data.cylinders.malloc(header.num.cylinderNum);
		::fread((r.cylinders.data.cylinders.data), header.num.cylinderNum, sizeof(RayTracing::Model::Cylinders::CylinderData::Data), temp);
	}
	if (header.num.coneNum)
	{
		::fseek(temp, header.offset.coneNum, SEEK_SET);
		r.cones.data.cones.malloc(header.num.coneNum);
		::fread((r.cones.data.cones.data), header.num.coneNum, sizeof(RayTracing::Model::Cones::ConeData::Data), temp);
	}
	if (header.num.pointLightNum)
	{
		::fseek(temp, header.offset.pointLightNum, SEEK_SET);
		r.pointLights.data.pointLights.malloc(header.num.pointLightNum);
		::fread((r.pointLights.data.pointLights.data), header.num.pointLightNum, sizeof(RayTracing::Model::PointLights::PointLightData::Data), temp);
	}
	::fclose(temp);
	return r;
}