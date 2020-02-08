#pragma once
#include <GL/_OpenGL.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>
#include <CUDA/_CUDA_NBody_Common.h>
#include <time.h>
#include <random>

namespace CUDA
{
	struct Buffer
	{
		enum BufferType
		{
			Device,
			GLinterop,
			ZeroCopy,
			Unused,
		};
		BufferType type;
		size_t size;
		size_t hostSize;
		cudaGraphicsResource* graphics;
		GLuint gl;
		void* device;
		void* host;

		Buffer(BufferType _type)
			:
			type(_type),
			size(0),
			hostSize(0),
			graphics(nullptr),
			gl(0),
			device(nullptr),
			host(nullptr)
		{
		}
		Buffer(BufferType _type, unsigned long long _size)
			:
			Buffer(_type)
		{
			size = _size;
			switch (_type)
			{
				case Device:
				case ZeroCopy:
				{
					resize(size_t(_size));
					break;
				}
				case GLinterop:
				{
					resize(GLuint(_size));
					break;
				}
			}
		}
		Buffer(GLuint _gl)
			:
			type(GLinterop),
			size(0),
			graphics(nullptr),
			device(nullptr),
			host(nullptr)
		{
			resize(_gl);
		}
		template<class T>Buffer(T const& a, bool copy)
			:
			type(Device),
			graphics(nullptr),
			device(nullptr),
			host(nullptr)
		{
			resize(sizeof(T));
			if (copy)cudaMemcpy(device, &a, size, cudaMemcpyHostToDevice);
		}
		~Buffer()
		{
			if (type != Unused)
				switch (type)
				{
					case Device:
					{
						freeHost();
						cudaFree(device);
						break;
					}
					case GLinterop:
					{
						unmap();
						freeHost();
						break;
					}
					case ZeroCopy:
					{
						cudaFreeHost(host);
						break;
					}
				}
			type = Unused;
			size = 0;
			hostSize = 0;
			graphics = nullptr;
			gl = 0;
			host = nullptr;
			device = nullptr;
		}
		void printInfo(char const* a)const
		{
			::printf("%s", a);
			::printf("[Type: ");
			switch (type)
			{
				case Device: ::printf("Device"); break;
				case GLinterop: ::printf("GLinterop"); break;
				case ZeroCopy: ::printf("ZeroCopy"); break;
				case Unused: ::printf("Unused"); break;
			}
			::printf(", Size: %llu, HostSize: %llu, GR: 0x%p, GL: %u, Device: 0x%p, Host: 0x%p]\n",
				size, hostSize, graphics, gl, device, host);
		}
		//Doesn't keep previous data...
		void resize(size_t _size)
		{
			size = _size;
			switch (type)
			{
				case Device:
				{
					cudaFree(device);
					cudaMalloc(&device, _size);
					break;
				}
				case ZeroCopy:
				{
					cudaFreeHost(host);
					cudaHostAlloc(&host, _size, cudaHostAllocPortable | cudaHostAllocMapped);
					cudaHostGetDevicePointer(&device, host, 0);
					break;
				}
				case GLinterop:break;
			}
		}
		void resize(GLuint _gl)
		{
			cudaGraphicsGLRegisterBuffer(&graphics, gl = _gl, cudaGraphicsRegisterFlagsNone);
			//map();
			//unmap();
		}
		void resizeHost()
		{
			if (size != hostSize)::free(host);
			if (size)host = ::malloc(hostSize = size);
		}
		void* map()
		{
			if (type == GLinterop)
			{
				cudaGraphicsMapResources(1, &graphics);
				cudaGraphicsResourceGetMappedPointer(&device, &size, graphics);
			}
			return device;
		}
		void unmap()
		{
			if (type == GLinterop)cudaGraphicsUnmapResources(1, &graphics);
			else cudaStreamSynchronize(0);
		}
		void freeHost()
		{
			::free(host);
			host = nullptr;
			hostSize = 0;
		}
		void moveToHost()
		{
			if (type == Device && size)
			{
				resizeHost();
				cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
			}
		}
		void moveToDevice()
		{
			if (type == Device && size && hostSize == size)
			{
				cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
			}
		}
		template<class T>void copy(T const& a)
		{
			if (size == 0 && type != GLinterop)resize(sizeof(T));
			if (size >= sizeof(T))
				cudaMemcpy(device, &a, sizeof(T), cudaMemcpyHostToDevice);
		}
		void copy(void* _src, size_t _size)
		{
			if (size == 0 && type != GLinterop)resize(_size);
			if (size >= _size)
				cudaMemcpy(device, _src, _size, cudaMemcpyHostToDevice);
		}
		void copy(Buffer& a)
		{
			type = a.type;
			size = a.size;
			graphics = a.graphics;
			gl = a.gl;
			device = a.device;
			host = a.host;
			a.type = Unused;
		}
		operator CUdeviceptr()const
		{
			return (CUdeviceptr)device;
		}
	};
	struct OpenGLDeviceInfo
	{
		unsigned int deviceCount;
		int devices[8];
		OpenGLDeviceInfo()
			:
			devices{ -1,-1,-1,-1,-1,-1,-1,-1 }
		{
			cudaGLGetDevices(&deviceCount, devices, 8, cudaGLDeviceListAll);
		}
		void printInfo()
		{
			::printf("Number of CUDA devices corresponding to the current OpenGL context:\t%u\n", deviceCount);
			::printf("Devices:\t");
			for (int c0(0); c0 < deviceCount; ++c0)
				::printf("%d\t", devices[c0]);
			::printf("\n");
		}
	};

	struct NBodyCUDA : ::OpenGL::OpenGL
	{
		struct Particles
		{
			struct Particle
			{
				Math::vec3<float>position;
				float mass;
				Math::vec4<float>velocity;
			};
			Vector<Particle>particles;
			std::mt19937 mt;
			std::uniform_real_distribution<float>randReal;
			unsigned int num;
			Particles() = delete;
			Particles(unsigned int _num)
				:
				mt(time(NULL)),
				num(_num),
				randReal(0, 1)
			{
			}
			Particle flatGalaxyParticles()
			{
				float r(100 * randReal(mt) + 0.1);
				float phi(2 * Math::Pi * randReal(mt));
				r = pow(r, 0.5);
				float vk(3.0f);
				float rn(0.3);
				return
				{
					{r * cos(phi),1.0f * randReal(mt) - 0.5f,r * sin(phi)},
					randReal(mt) > 0.999f ? 100 : randReal(mt),
					{-vk * sin(phi) / powf(r,rn),0,vk * cos(phi) / powf(r,rn)},
				};
			}
			Particle flatGalaxyParticlesOptimized(float blackHoleMass)
			{
				float r0(sqrtf(randReal(mt) + 0.01));
				float phi(2 * Math::Pi * randReal(mt));
				float r = r0 * 5;
				float vk(sqrtf(0.001f * (r * calcForce(r0) + blackHoleMass / r)));
				return
				{
					{r * cos(phi),1.0f * randReal(mt) - 0.5f,r * sin(phi)},
					randReal(mt),
					{-vk * sin(phi) ,0,vk * cos(phi) },
				};
			}
			Particle sphereGalaxyParticles()
			{
				float r(pow(100.0f * randReal(mt) + 0.1f, 1.0 / 3));
				float theta(2.0f * acos(randReal(mt)));
				float phi(2 * Math::Pi * randReal(mt));
				float vk(1.7f);
				float rn(0.5);
				return
				{
					{r * cos(phi) * sin(theta),r * sin(phi) * sin(theta),r * cos(theta)},
					randReal(mt) > 0.999f ? 100 : randReal(mt),
					{-vk * sin(phi) / powf(r,rn),vk * cos(phi) / powf(r,rn),0},
				};
			}
			Particle expFlatGalaxyParticles()
			{
				float r(100 * randReal(mt));
				float phi(2 * Math::Pi * randReal(mt));
				r = pow(r, 0.5);
				float vk(3.0f);
				float rn(0.3);
				return
				{
					{r * cos(phi),r * sin(phi),1.0f * randReal(mt) - 0.5f},
					randReal(mt),
					{-vk * sin(phi) / powf(r,rn),vk * cos(phi) / powf(r,rn),0},
				};
			}
			float calcForce(float r)
			{
				//r is in [0, 1], mass is uniformly distrubuted in [0, 1]
				return (0.00434f + r * (-0.03039f +
					r * (0.11616f + r * (-0.16195f + 0.08362f * r)))) * num;
			}
			void experimentGalaxy()
			{
				//This is used to create a distrubution without center black hole...
				//to see how force is distrubuted.
				unsigned int _num(num);
				while (_num--)particles.pushBack(expFlatGalaxyParticles());
			}
			void randomGalaxy()
			{
				unsigned int _num(num - 1);
				while (_num--)
					particles.pushBack(flatGalaxyParticles());
				particles.pushBack
				(
					{
						{0,0,0},
						8000,
						{0,0,0},
					}
				);
			}
			void randomGalaxyOptimized()
			{
				unsigned int _num(num - 1);
				float blackHoleMass(200000.0f);
				while (_num--)
					particles.pushBack(flatGalaxyParticlesOptimized(blackHoleMass));
				particles.pushBack
				(
					{
						{0,0,0},
						blackHoleMass,
						{0,0,0},
					}
				);
			}
		};
		struct ParticlesData : ::OpenGL::Buffer::Data
		{
			Particles* particles;
			ParticlesData(Particles* _particles)
				:
				Data(DynamicDraw),
				particles(_particles)
			{
			}
			virtual void* pointer()override
			{
				return particles->particles.data;
			}
			virtual unsigned int size()override
			{
				return sizeof(Particles::Particle) * (particles->particles.length);
			}
		};
		struct Renderer : ::OpenGL::Program
		{
			::OpenGL::Buffer transBuffer;
			::OpenGL::BufferConfig transUniform;
			::OpenGL::BufferConfig particlesArray;
			::OpenGL::VertexAttrib positions;
			::OpenGL::VertexAttrib velocities;

			Renderer(::OpenGL::SourceManager* _sm, ::OpenGL::Buffer* _particlesBuffer, ::OpenGL::Transform* _trans)
				:
				Program(_sm, "Renderer", Vector< ::OpenGL::VertexAttrib*>{&positions, & velocities}),
				transBuffer(&_trans->bufferData),
				transUniform(&transBuffer, ::OpenGL::UniformBuffer, 0),
				particlesArray(_particlesBuffer, ::OpenGL::ArrayBuffer),
				positions(&particlesArray, 0, ::OpenGL::VertexAttrib::three, ::OpenGL::VertexAttrib::Float, false, sizeof(Particles::Particle), 0, 0),
				velocities(&particlesArray, 1, ::OpenGL::VertexAttrib::three, ::OpenGL::VertexAttrib::Float, false, sizeof(Particles::Particle), 16, 0)
			{
				init();
			}
			virtual void initBufferData()override
			{
			}
			virtual void run()override
			{
				glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				glDrawArrays(GL_POINTS, 0, particlesArray.buffer->data->size() / sizeof(Particles::Particle));
			}
		};

		::OpenGL::SourceManager sm;
		Particles particles;
		ParticlesData particlesData;
		::OpenGL::Buffer particlesBuffer;
		::OpenGL::Transform trans;
		Renderer renderer;
		Buffer particlesBufferCUDA;
		NBodyCUDA_Glue glue;


		NBodyCUDA(unsigned int _blocks, bool _experiment)
			:
			sm(),
			particles(_blocks * 1024),
			particlesData(&particles),
			particlesBuffer(&particlesData),
			trans({ {80.0,0.1,800},{0.8,0.8,0.1},{1},500.0 }),
			renderer(&sm, &particlesBuffer, &trans),
			particlesBufferCUDA(Buffer::GLinterop),
			glue(_blocks, 0.005f, 0.001f)
		{
			if (_experiment)
				particles.experimentGalaxy();
			else
				particles.randomGalaxyOptimized();
		}
		void experiment()
		{
			Buffer expBuffer(Buffer::Device);
			expBuffer.resize(particles.particles.length * sizeof(ExpData));
			glue.experiment((ExpData*)expBuffer.device);
			expBuffer.moveToHost();
			File file("./");
			String<char> answer;
			for (int c0(0); c0 < particles.particles.length; ++c0)
			{
				char tp[50];
				ExpData& data(((ExpData*)expBuffer.host)[c0]);
				sprintf(tp, "%f %f\n", data.r, data.force);
				answer += tp;
			}
			file.createText("answer.txt", answer);
		}
		virtual void init(::OpenGL::FrameScale const& _size)override
		{
			glViewport(0, 0, _size.w, _size.h);
			glPointSize(2);
			glEnable(GL_DEPTH_TEST);
			trans.init(_size);
			renderer.transUniform.dataInit();
			renderer.particlesArray.dataInit();
			particlesBufferCUDA.resize(renderer.particlesArray.buffer->buffer);
			glue.particles = (NBodyCUDAParticle*)particlesBufferCUDA.map();
		}
		virtual void run()override
		{
			trans.operate();
			if (trans.updated)
			{
				renderer.transUniform.refreshData();
				trans.updated = false;
			}
			renderer.use();
			renderer.run();
			glFinish();
			glue.run();
		}
		virtual void frameSize(int _w, int _h) override
		{
			trans.resize(_w, _h);
			glViewport(0, 0, _w, _h);
		}
		virtual void framePos(int, int) override
		{
		}
		virtual void frameFocus(int) override
		{
		}
		virtual void mouseButton(int _button, int _action, int _mods) override
		{
			switch (_button)
			{
				case GLFW_MOUSE_BUTTON_LEFT:trans.mouse.refreshButton(0, _action); break;
				case GLFW_MOUSE_BUTTON_MIDDLE:trans.mouse.refreshButton(1, _action); break;
				case GLFW_MOUSE_BUTTON_RIGHT:trans.mouse.refreshButton(2, _action); break;
			}
		}
		virtual void mousePos(double _x, double _y) override
		{
			trans.mouse.refreshPos(_x, _y);
		}
		virtual void mouseScroll(double _x, double _y)override
		{
			if (_y != 0.0)
				trans.scroll.refresh(_y);
		}
		virtual void key(GLFWwindow* _window, int _key, int _scancode, int _action, int _mods) override
		{
			switch (_key)
			{
				case GLFW_KEY_ESCAPE:
					if (_action == GLFW_PRESS)
					{
						particlesBufferCUDA.unmap();
						glfwSetWindowShouldClose(_window, true);
					}
					break;
				case GLFW_KEY_A:trans.key.refresh(0, _action); break;
				case GLFW_KEY_D:trans.key.refresh(1, _action); break;
				case GLFW_KEY_W:trans.key.refresh(2, _action); break;
				case GLFW_KEY_S:trans.key.refresh(3, _action); break;
			}
		}
	};
}