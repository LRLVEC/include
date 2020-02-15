#pragma once

#include <GL/_OpenGL.h>
#include <GL/_Texture.h>
#include <GL/_OpenVR.h>
#include <_Math.h>
#include <_Pair.h>
#include <_Array.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <CUDA/_CUDA.h>


namespace OpenGL
{
	struct OptiXDefautRenderer : ::OpenGL::Program
	{
		struct TriangleData : ::OpenGL::Buffer::Data
		{
			using Vertex = Math::vec2<float>;
			using Triangle = Array<Vertex, 3>;
			Array<Triangle, 2> triangles;
			TriangleData()
				:
				Data(StaticDraw),
				triangles({ {{-1,-1},{1,-1},{1,1}},{{1,1},{-1,1},{-1,-1}} })
			{
			}
			virtual void* pointer()override
			{
				return (void*)triangles.data;
			}
			virtual unsigned int size()override
			{
				return sizeof(Triangle) * triangles.length;
			}
		};
		struct PixelData : ::OpenGL::Buffer::Data
		{
			Texture frameTexture;
			TextureConfig<TextureStorage2D> frameConfig;

			PixelData(FrameScale const& _size)
				:
				Data(StreamDraw),
				frameTexture(nullptr, 0),
				frameConfig(&frameTexture, Texture2D, RGBA32f, 1, _size.w, _size.h)
			{
			}
			virtual void* pointer()override
			{
				return nullptr;
			}
			virtual unsigned int size()override
			{
				return frameConfig.width * frameConfig.height * 16;
			}
		};

		TriangleData triangles;
		PixelData pixelData;
		::OpenGL::Buffer trianglesBuffer;
		::OpenGL::Buffer pixelBuffer;
		BufferConfig bufferArray;
		BufferConfig pixelPixel;
		VertexAttrib positions;
		bool updated;

		OptiXDefautRenderer(SourceManager* _sourceManage, FrameScale const& _size)
			:
			Program(_sourceManage, "Frame", Vector<VertexAttrib*>{&positions}),
			triangles(),
			pixelData(_size),
			trianglesBuffer(&triangles),
			pixelBuffer(&pixelData),
			bufferArray(&trianglesBuffer, ArrayBuffer),
			pixelPixel(&pixelBuffer, PixelUnpackBuffer),
			positions(&bufferArray, 0, VertexAttrib::two,
				VertexAttrib::Float, false, sizeof(TriangleData::Vertex), 0, 0),
			updated(false)
		{
			init();
			prepare();
		}
		virtual void initBufferData()override
		{
		}
		virtual void run() override
		{
			if (updated)
			{
				pixelPixel.bind();
				pixelData.frameConfig.dataInit(0, TextureInputRGBA, TextureInputFloat);
				updated = false;
				pixelPixel.unbind();
			}
			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT);
			glDrawArrays(GL_TRIANGLES, 0, 6);
		}
		void prepare()
		{
			bufferArray.dataInit();
			use();
			pixelData.frameTexture.bindUnit();
			pixelPixel.dataInit();
		}
		void resize(FrameScale const& _size)
		{
			glViewport(0, 0, _size.w, _size.h);
			pixelData.frameConfig.resize(_size.w, _size.h);
			pixelPixel.dataInit();
		}
		FrameScale size()const
		{
			return{ int(pixelData.frameConfig.width), int(pixelData.frameConfig.height) };
		}
		operator GLuint ()const
		{
			return pixelBuffer.buffer;
		}
	};

	namespace OptiX
	{
		template <class T>struct SbtRecord
		{
			__declspec(align(16)) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
			T data;
		};
		struct Context
		{
			CUcontext cudaContext;
			OptixDeviceContext optixContext;
			Context()
				:
				cudaContext(0)
			{
				cudaFree(0);
				optixInit();
				OptixDeviceContextOptions options{ nullptr,nullptr,0 };
				optixDeviceContextCreate(cudaContext, &options, &optixContext);
			}
			~Context()
			{
				optixDeviceContextDestroy(optixContext);
			}
			operator OptixDeviceContext()const
			{
				return optixContext;
			}
		};
		struct ModuleManager
		{
			struct Module
			{
				String<char>name;
				Vector<String<char>>functions;
				OptixModule module;
				Module(String<char>const& _name, String<char>const& _source,
					OptixDeviceContext _oc, OptixModuleCompileOptions* _mco,
					OptixPipelineCompileOptions* _pco)
					:
					name(_name)
				{
					optixModuleCreateFromPTX(_oc, _mco, _pco, _source,
						_source.length, nullptr, nullptr, &module);
				}
				bool operator==(String<char> const& _name)const
				{
					for (int c0(0); c0 < functions.length; ++c0)
						if (_name == functions.data[c0])return true;
				}
			};
			File* folder;
			Vector<Module>modules;

			ModuleManager(File* _folder, OptixDeviceContext _oc,
				OptixModuleCompileOptions* _mco, OptixPipelineCompileOptions* _pco)
				:
				folder(_folder)
			{
				readSource(_oc, _mco, _pco);
			}
			void readSource(OptixDeviceContext _oc, OptixModuleCompileOptions* _mco,
				OptixPipelineCompileOptions* _pco)
			{
				File& ptx(folder->find("ptx"));
				String<char>moduleLists(ptx.find("ModuleLists.txt").readText());
				Vector<int>files(moduleLists.find("Program:"));
				for (int c0(0); c0 < files.length; ++c0)
				{
					int n(files[c0]);
					int delta;
					char t[100];
					char t1[10];
					n += sweepStr(moduleLists.data + n, "Program:%*[^a-zA-Z0-9_]%n");
					sscanf(moduleLists.data + n, "%[a-zA-Z0-9_]%n", t, &delta);
					n += delta;
					n += sweepStr(moduleLists.data + n, "%*[\r\n{]%n");
					String<char>name(t);
					Module s(name, ptx.findInThis(name + ".ptx").readText(),
						_oc, _mco, _pco);
					n += sweepStr(moduleLists.data + n, "%*[^{]%{%n");
					int j(0);
					do
					{
						j = sscanf(moduleLists.data + n, "%s%n%*[\t\r\n ]%[}]", t, &delta, t1);
						n += delta;
						if (j < 1)break;
						s.functions.pushBack(t);
					} while (j == 1);
					modules.pushBack(s);
				}
			}
			OptixModule getModule(String<char>const& _programName)const
			{
				for (int c0(0); c0 < modules.length; ++c0)
					if (modules.data[c0] == _programName)
						return modules.data[c0].module;
			}
		};
		struct Program
		{
			enum ProgramKind
			{
				RayGen = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
				Miss = OPTIX_PROGRAM_GROUP_KIND_MISS,
				Exception = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION,
				HitGroup = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
				Callables = OPTIX_PROGRAM_GROUP_KIND_CALLABLES
			};
			Vector<String<char>>name;
			ProgramKind kind;
			OptixProgramGroup program;
			OptixDeviceContext optixContext;
			OptixProgramGroupOptions* options;
			Program(Vector<String<char>>const& _name, ProgramKind _kind, OptixProgramGroupOptions* _options,
				OptixDeviceContext _optixContext, ModuleManager* moduleManager)
				:
				name(_name),
				kind(_kind),
				optixContext(_optixContext),
				options(_options)
			{
				OptixProgramGroupDesc desc{};
				desc.kind = (OptixProgramGroupKind)_kind;
				if (_name.length)switch (kind)
				{
					case RayGen:
					{
						desc.raygen.entryFunctionName = _name.data[0].data;
						desc.raygen.module = moduleManager->getModule(_name.data[0]);
						break;
					}
					case Miss:
					{
						desc.miss.entryFunctionName = _name.data[0].data;
						desc.miss.module = moduleManager->getModule(_name.data[0]);
						break;
					}
					case Exception:
					{
						desc.exception.entryFunctionName = _name.data[0].data;
						desc.exception.module = moduleManager->getModule(_name.data[0]);
						break;
					}
					case HitGroup:
					{
						desc.hitgroup.entryFunctionNameCH = _name.data[0].data;
						if (_name.length > 1)desc.hitgroup.entryFunctionNameAH = _name.data[1].data;
						if (_name.length > 2)desc.hitgroup.entryFunctionNameIS = _name.data[2].data;
						desc.hitgroup.moduleCH = moduleManager->getModule(_name.data[0]);
						if (_name.length > 1)desc.hitgroup.moduleAH = moduleManager->getModule(_name.data[1]);
						if (_name.length > 2)desc.hitgroup.moduleIS = moduleManager->getModule(_name.data[2]);
						break;
					}
					case Callables:
					{
						desc.callables.entryFunctionNameDC = _name.data[0].data;
						if (_name.length > 1)desc.callables.entryFunctionNameCC = _name.data[1].data;
						desc.callables.moduleDC = moduleManager->getModule(_name.data[0]);
						if (_name.length > 1)desc.callables.moduleCC = moduleManager->getModule(_name.data[1]);
						break;
					}
				}
				optixProgramGroupCreate(optixContext, &desc, 1, _options, nullptr, nullptr, &program);
			}
			operator OptixProgramGroup()const
			{
				return program;
			}
		};
		struct Pipeline
		{
			OptixPipeline pipeline;
			Pipeline(OptixDeviceContext _oc, OptixPipelineCompileOptions* _pco,
				OptixPipelineLinkOptions* _plo, Vector<OptixProgramGroup>const& _programs)
			{
				optixPipelineCreate(_oc, _pco, _plo, _programs.data,
					_programs.length, nullptr, nullptr, &pipeline);
			}
			operator OptixPipeline()const
			{
				return pipeline;
			}
		};
		struct Trans
		{
			struct Data
			{
				struct Perspective
				{
					double fovy;
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
					fovy(Math::Pi * 100.0 / 180.0),
					y(1024),
					updated(false)
				{
				}
				Perspective(Data::Perspective const& _persp)
					:
					fovy(_persp.fovy),
					y(1024),
					updated(false)
				{
				}
				Perspective(Perspective const&) = default;
				void init(::OpenGL::FrameScale const& _size)
				{
					y = _size.h;
					updated = false;
				}
				void refresh(::OpenGL::FrameScale const& _size)
				{
					y = _size.h;
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
			struct TransInfo
			{
				Math::mat<float, 3, 4>ans;
				Math::vec3<float>r0;
				float z0;
			};

			Perspective persp;
			Scroll scroll;
			Key key;
			Mouse mouse;
			CUDA::Buffer buffer;
			Math::vec3<double>dr;
			Math::mat3<double>trans;
			double depth;
			bool moved;
			bool updated;

			Trans() = delete;
			Trans(Data const& _data)
				:
				persp(_data.persp),
				scroll(_data.scroll),
				key(_data.key),
				mouse(),
				moved(true),
				updated(false),
				buffer(CUDA::Buffer::Device, sizeof(TransInfo)),
				dr(_data.initialPosition),
				trans(Math::mat3<double>::id()),
				depth(_data.depth)
			{
			}
			void init(::OpenGL::FrameScale const& _size)
			{
				buffer.resizeHost();
				persp.init(_size);
				((TransInfo*)(buffer.host))->z0 = -float(_size.h) / tan(Math::Pi * persp.fovy / 360.0);
				calcAns();
				updated = true;
				buffer.moveToDevice();
			}
			void resize(::OpenGL::FrameScale const& _size)
			{
				persp.refresh(_size);
			}
			void calcAns()
			{
				((TransInfo*)(buffer.host))->ans = trans;
				((TransInfo*)(buffer.host))->r0 = dr;
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
					((TransInfo*)(buffer.host))->z0 = -(persp.y / tan(Math::Pi * persp.fovy / 360.0));
					persp.updated = false;
					operated = true;
				}
				if (operated)
				{
					calcAns();
					buffer.moveToDevice();
					updated = true;
				}
			}
		};
		struct RayTracer
		{
			virtual void run() = 0;
			virtual void resize(FrameScale const& _size, GLuint _gl) = 0;
		};
	}

	namespace VR
	{
		struct OptiXTrans
		{

		};
	}
}