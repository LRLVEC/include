#pragma once

#include <GL/_OpenVR.h>
#include <GL/_Texture.h>
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
	struct OptiXRenderer : Program
	{
		struct TriangleData : Buffer::Data
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
		struct PixelData : Buffer::Data
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
		Buffer trianglesBuffer;
		Buffer pixelBuffer;
		BufferConfig bufferArray;
		BufferConfig pixelPixel;
		VertexAttrib positions;

		OptiXRenderer(SourceManager* _sourceManager, FrameScale const& _size)
			:
			Program(_sourceManager, "Frame", Vector<VertexAttrib*>{&positions}),
			triangles(),
			pixelData(_size),
			trianglesBuffer(&triangles),
			pixelBuffer(&pixelData),
			bufferArray(&trianglesBuffer, ArrayBuffer),
			pixelPixel(&pixelBuffer, PixelUnpackBuffer),
			positions(&bufferArray, 0, VertexAttrib::two,
				VertexAttrib::Float, false, sizeof(TriangleData::Vertex), 0, 0)
		{
		}
		operator GLuint ()const
		{
			return pixelBuffer.buffer;
		}
		FrameScale size()const
		{
			return{ int(pixelData.frameConfig.width), int(pixelData.frameConfig.height) };
		}
	};
	struct OptiXDefautRenderer : OptiXRenderer
	{
		bool updated;
		OptiXDefautRenderer(SourceManager* _sourceManager, FrameScale const& _size)
			:
			OptiXRenderer(_sourceManager, _size),
			updated(false)
		{
			init();
			prepare();
		}
		void prepare()
		{
			bufferArray.dataInit();
			use();
			pixelData.frameTexture.bindUnit();
			pixelPixel.dataInit();
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
		void resize(FrameScale const& _size)
		{
			glViewport(0, 0, _size.w, _size.h);
			pixelData.frameConfig.resize(_size.w, _size.h);
			pixelPixel.dataInit();
		}
	};
	namespace VR
	{
		void Object::updateOptiX(vr::TrackedDevicePose_t const& a)
		{
			float const(*m)[4](a.mDeviceToAbsoluteTracking.m);
			pos.array[0][0] = m[0][0]; pos.array[0][1] = m[0][1]; pos.array[0][2] = m[0][2];
			pos.array[1][0] = m[1][0]; pos.array[1][1] = m[1][1]; pos.array[1][2] = m[1][2];
			pos.array[2][0] = m[2][0]; pos.array[2][1] = m[2][1]; pos.array[2][2] = m[2][2];
			pos.array[3][0] = m[0][3]; pos.array[3][1] = m[1][3]; pos.array[3][2] = m[2][3];

			velocity = *(Math::vec3<float>*) & a.vVelocity;
			omega = *(Math::vec3<float>*) & a.vAngularVelocity;
			trackingResult = a.eTrackingResult;
			posValid = a.bPoseIsValid;
			connected = a.bDeviceIsConnected;
		}
		struct OptiXTrans
		{
			struct TransInfo
			{
				Math::mat<float, 3, 4>ans;
				Math::vec3<float>r0;
				float z0;
			};
			struct Perspective
			{
				double zNear;
				double zFar;
			};
			struct SingleEye
			{
				vr::EVREye eye;
				VRDevice* hmd;
				Perspective* persp;
				float d0;
				Math::vec2<float> rayOffset;
				//Normally doesn't change until you change the distance between two eyes.
				//In fact, we can use a vec3<float> instead because it's just an offset from eye to head in head space.
				//Math::mat4<float> r0;
				Math::vec3<float> eyeOffset;
				Math::mat4<float> trans;

				SingleEye() = delete;
				SingleEye(VRDevice* _hmd, vr::EVREye _eye, Perspective* _persp)
					:
					hmd(_hmd),
					eye(_eye),
					persp(_persp)
				{
				}
				void updateProj()
				{
					vr::HmdMatrix44_t projMat4 = hmd->hmd->GetProjectionMatrix(eye, persp->zNear, persp->zFar);
					d0 = hmd->frameScale.w * projMat4.m[0][0] / 2.0f;
					rayOffset = { hmd->frameScale.w * projMat4.m[0][2]  ,hmd->frameScale.h * projMat4.m[1][2] };
					rayOffset /= 2.0f;
					//::printf("d0: %f\nrayOffset: [%f, %f]\n", d0, rayOffset.data[0], rayOffset.data[1]);
				}
				void updateOffset()
				{
					vr::HmdMatrix34_t offsetMat4 = hmd->hmd->GetEyeToHeadTransform(eye);
					Math::mat<float, 3, 4>m0(*(Math::mat<float, 3, 4>*) & offsetMat4);
					eyeOffset = { offsetMat4.m[0][3],offsetMat4.m[1][3], offsetMat4.m[2][3] };
				}
				void updateTrans(Object const& _hmd)
				{
					trans = _hmd.pos;
					trans.rowVec[3] += (_hmd.pos, eyeOffset);
					trans.array[3][3] = -d0;
					trans.array[0][3] = rayOffset.data[0];
					trans.array[1][3] = rayOffset.data[1];
				}
				void updateAll()//if proj or offset changes, use this
				{
					updateProj();
					updateOffset();
					updateTrans(hmd->objects[0]);
				}
				void update()
				{
					updateTrans(hmd->objects[0]);
				}
				void printInfo()const
				{
					::printf(eye == vr::Eye_Left ? "Left eye:\n" : "Right eye:\n");
					eyeOffset.printInfo("Offset: ");
					trans.printInfo("\nTrans: ");
				}
			};
			CUDA::Buffer buffer;
			VRDevice* hmd;
			Perspective persp;
			SingleEye leftEye;
			SingleEye rightEye;

			OptiXTrans() = delete;
			OptiXTrans(VRDevice* _hmd, Perspective _persp)
				:
				hmd(_hmd),
				persp(_persp),
				leftEye(_hmd, vr::Eye_Left, &persp),
				rightEye(_hmd, vr::Eye_Right, &persp),
				buffer(CUDA::Buffer::Device, sizeof(TransInfo))
			{
				updateAll();
			}
			void update()
			{
				hmd->refreshHMDOptiX();
				leftEye.update();
				rightEye.update();
			}
			void updateAll()
			{
				hmd->refreshHMDOptiX();
				leftEye.updateAll();
				rightEye.updateAll();
			}
			void operate(bool isRightEye)
			{
				if (!isRightEye)buffer.copy(leftEye.trans);
				else buffer.copy(rightEye.trans);
			}
		};
		struct OptiXVRRenderer : OptiXRenderer
		{
			VRDevice* hmd;
			FrameBufferDesc leftEyeDesc;
			FrameBufferDesc rightEyeDesc;
			FrameScale windowSize;

			OptiXVRRenderer(SourceManager* _sourceManager, FrameScale const& _size, VRDevice* _hmd)
				:
				OptiXRenderer(_sourceManager, _hmd->frameScale),
				hmd(_hmd),
				leftEyeDesc(hmd->frameScale),
				rightEyeDesc(hmd->frameScale),
				windowSize(_size)
			{
				init();
				prepare();
				//glDisable(GL_DEPTH_TEST);
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
				windowSize = _size;
			}
			virtual void initBufferData()override
			{

			}
			virtual void run()override
			{
			}
			void refreshFrameData()
			{
				pixelPixel.bind();
				pixelData.frameConfig.dataInit(0, TextureInputRGBA, TextureInputFloat);
				pixelPixel.unbind();
			}
			void renderLeft()
			{
				glBindFramebuffer(GL_FRAMEBUFFER, leftEyeDesc.renderFramebuffer);
				glViewport(0, 0, hmd->frameScale.w, hmd->frameScale.h);
				pixelData.frameConfig.bind();
				glDrawArrays(GL_TRIANGLES, 0, 6);
				leftEyeDesc.copyRenderBuffer();

				glBindFramebuffer(GL_READ_FRAMEBUFFER, leftEyeDesc.renderFramebuffer);
				glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
				glBlitFramebuffer(0, 0, leftEyeDesc.size.w, leftEyeDesc.size.h,
					0, 0, leftEyeDesc.size.w, leftEyeDesc.size.h, GL_COLOR_BUFFER_BIT, GL_LINEAR);
				glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
				glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

			}
			void renderRight()
			{
				glBindFramebuffer(GL_FRAMEBUFFER, rightEyeDesc.renderFramebuffer);
				pixelData.frameConfig.bind();
				glDrawArrays(GL_TRIANGLES, 0, 6);
				rightEyeDesc.copyRenderBuffer();

				glBindFramebuffer(GL_READ_FRAMEBUFFER, rightEyeDesc.renderFramebuffer);
				glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
				glBlitFramebuffer(0, 0, rightEyeDesc.size.w, rightEyeDesc.size.h,
					rightEyeDesc.size.w, 0, 2 * leftEyeDesc.size.w, rightEyeDesc.size.h, GL_COLOR_BUFFER_BIT, GL_LINEAR);
				glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
				glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
			}
			void renderWindow()
			{
				glBindFramebuffer(GL_FRAMEBUFFER, 0);
				glViewport(0, 0, windowSize.w, windowSize.h);
				pixelData.frameConfig.bind();
				glDrawArrays(GL_TRIANGLES, 0, 6);
			}
			void commit()
			{
				vr::Texture_t leftEyeTexture = { (void*)(uintptr_t)leftEyeDesc.resolveTexture,
					vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
				vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTexture);
				vr::Texture_t rightEyeTexture = { (void*)(uintptr_t)rightEyeDesc.resolveTexture,
					vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
				vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTexture);
				glFlush();
			}
		};
	}
}

namespace CUDA
{
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
					char tp[1024];
					size_t sa(1024);
					optixModuleCreateFromPTX(_oc, _mco, _pco, _source,
						_source.length, tp, &sa, &module);
					::printf("optixModuleCreateFromPTX: %s\n", tp);
				}
				bool operator==(String<char> const& _name)const
				{
					for (int c0(0); c0 < functions.length; ++c0)
						if (_name == functions.data[c0])return true;
					return false;
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
				char tp[1024];
				size_t sa(1024);
				optixProgramGroupCreate(optixContext, &desc, 1, _options, tp, &sa, &program);
				::printf("Program Create (%s): %s\n", _name.data[0].data, tp);
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
				char tp[1024];
				size_t sa(1024);
				optixPipelineCreate(_oc, _pco, _plo, _programs.data,
					_programs.length, tp, &sa, &pipeline);
				::printf("Pipeline Create: %s", tp);

				OptixStackSizes stack_sizes = {};
				for (unsigned int c0(0); c0 < _programs.length; ++c0)
					optixUtilAccumulateStackSizes(_programs.data[c0], &stack_sizes);

				uint32_t direct_callable_stack_size_from_traversal;
				uint32_t direct_callable_stack_size_from_state;
				uint32_t continuation_stack_size;
				//...
				optixUtilComputeStackSizes(&stack_sizes, _plo->maxTraceDepth,
					0,  // maxCCDepth
					0,  // maxDCDEpth
					&direct_callable_stack_size_from_traversal,
					&direct_callable_stack_size_from_state, &continuation_stack_size);
				optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
					direct_callable_stack_size_from_state, continuation_stack_size,
					1  // maxTraversableDepth
				);
			}
			operator OptixPipeline()const
			{
				return pipeline;
			}
		};
		struct Denoiser
		{
			OptixDenoiser denoiser;
			OptixDenoiserParams params;
			OpenGL::FrameScale size;
			OptixImage2D inputs[3];
			OptixImage2D output;
			CUdeviceptr scratch;
			CUdeviceptr intensity;
			CUdeviceptr state;
			unsigned int scratchSize;
			unsigned int stateSize;

			Denoiser(OptixDeviceContext _oc, OptixDenoiserOptions* _do, OptixDenoiserModelKind _dmk,
				OpenGL::FrameScale const& _size)
				:
				denoiser(nullptr),
				params{},
				size(_size),
				inputs{},
				output()
			{
				optixDenoiserCreate(_oc, _do, &denoiser);
				optixDenoiserSetModel(denoiser, _dmk, nullptr, 0);
				OptixDenoiserSizes denoiserSizes;
				optixDenoiserComputeMemoryResources(denoiser, _size.w, _size.h, &denoiserSizes);
				scratchSize = denoiserSizes.withoutOverlapScratchSizeInBytes;//default: not using tiled denoising
				stateSize = denoiserSizes.stateSizeInBytes;
				cudaMalloc((void**)&intensity, sizeof(float));
				cudaMalloc((void**)&state, stateSize);
				cudaMalloc((void**)&scratch, scratchSize);
			}
			void setup(float* inputRGB, float* inputAlbedo, float* inputNormal, CUstream cuStream)
			{
				inputs[0].data = (CUdeviceptr)inputRGB;
				inputs[0].width = size.w;
				inputs[0].height = size.h;
				inputs[0].rowStrideInBytes = size.w * sizeof(float4);
				inputs[0].pixelStrideInBytes = sizeof(float4);
				inputs[0].format = OPTIX_PIXEL_FORMAT_FLOAT4;

				inputs[1].data = 0;
				if (inputAlbedo)
				{
					inputs[1].data = (CUdeviceptr)inputAlbedo;
					inputs[1].width = size.w;
					inputs[1].height = size.h;
					inputs[1].rowStrideInBytes = size.w * sizeof(float4);
					inputs[1].pixelStrideInBytes = sizeof(float4);
					inputs[1].format = OPTIX_PIXEL_FORMAT_FLOAT4;
				}

				inputs[2].data = 0;
				if (inputAlbedo)
				{
					inputs[2].data = (CUdeviceptr)inputNormal;
					inputs[2].width = size.w;
					inputs[2].height = size.h;
					inputs[2].rowStrideInBytes = size.w * sizeof(float4);
					inputs[2].pixelStrideInBytes = sizeof(float4);
					inputs[2].format = OPTIX_PIXEL_FORMAT_FLOAT4;
				}
				optixDenoiserSetup(denoiser, cuStream, size.w, size.h,
					state, stateSize, scratch, scratchSize);
				params.denoiseAlpha = 0;
				params.hdrIntensity = intensity;
				params.blendFactor = 0.f;
			}
			void run(CUstream cuStream)
			{
				optixDenoiserComputeIntensity(denoiser, cuStream, inputs, intensity, scratch, scratchSize);
				optixDenoiserInvoke(denoiser, cuStream, &params, state, stateSize,
					inputs, inputs[2].data ? 3 : inputs[1].data ? 2 : 1,
					0, 0, &output, scratch, scratchSize);
			}
			~Denoiser()
			{
				cudaFree((void*)intensity);
				cudaFree((void*)state);
				cudaFree((void*)scratch);
				optixDenoiserDestroy(denoiser);
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
			Buffer buffer;
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
				buffer(Buffer::Device, sizeof(TransInfo)),
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
			virtual void resize(OpenGL::FrameScale const& _size, GLuint _gl) = 0;
		};
	}
}