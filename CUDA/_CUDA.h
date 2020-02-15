#pragma once
#include <GL/_OpenGL.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>
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
			if (_size)
			{
				if (size >= _size)cudaMemcpy(device, _src, _size, cudaMemcpyHostToDevice);
			}
			else cudaMemcpy(device, _src, size, cudaMemcpyHostToDevice);
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
}