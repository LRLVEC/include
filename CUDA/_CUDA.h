#pragma once
#include <GL/_OpenGL.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>
#include <time.h>
#include <random>
#include <_BMP.h>

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
			hostSize(0),
			graphics(nullptr),
			device(nullptr),
			host(nullptr)
		{
			resize(_gl);
		}
		template<class T>Buffer(T const& a, bool copy)
			:
			type(Device),
			size(0),
			hostSize(0),
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
			//bug here!!!!!
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
			if (type == GLinterop)
			{
				cudaGraphicsUnmapResources(1, &graphics);
				device = nullptr;
			}
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
			if (host && device)
			{
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
		void clearDevice(int val)
		{
			if (device)cudaMemset(device, val, size);
		}
		operator CUdeviceptr()const
		{
			return (CUdeviceptr)device;
		}
	};
	struct CubeMap
	{
		BMP::Pixel_32* data;
		unsigned int width;

		CubeMap() :data(nullptr), width(0) {}
		CubeMap(String<char>const& _path) :data(nullptr), width(0)
		{
			String<char> names[6]{ "right.bmp","left.bmp" ,"top.bmp" ,"bottom.bmp"  ,"back.bmp","front.bmp" };
			BMP tp(_path + names[0], true);
			width = tp.header.width;
			size_t sz(sizeof(BMP::Pixel_32) * width * width);
			data = (BMP::Pixel_32*)malloc(6 * sz);
			memcpy(data, tp.data_32, sz);
			for (int c0(1); c0 < 6; ++c0)
			{
				BMP ts(_path + names[c0], true);
				memcpy(data + c0 * sz / 4, ts.data_32, sz);
			}
		}
		~CubeMap()
		{
			::free(data);
			data = nullptr;
		}
		void moveToGPU(cudaArray* _cuArray)const
		{
			cudaMemcpy3DParms cpy3Dparams
			{
				nullptr,{0,0,0},{data, width * 4ll,width, width},
				_cuArray,{0,0,0},{0},{ width, width, 6 }, cudaMemcpyHostToDevice
			};
			cudaMemcpy3D(&cpy3Dparams);
		}
	};
	template<unsigned int dim>struct Texture
	{
		static_assert(dim&& dim < 4, "Dim must be one of 1, 2, 3!");
	};
	template<>struct Texture<1>
	{
		cudaArray* data;
		cudaTextureObject_t textureObj;
		Texture(cudaChannelFormatDesc const& _cd, cudaTextureAddressMode _am,
			cudaTextureFilterMode _fm, cudaTextureReadMode _rm, bool normalizedCoords, size_t width)
			:
			data(nullptr),
			textureObj(0)
		{
			if (width)
			{
				cudaMallocArray(&data, &_cd, 256);
				cudaResourceDesc resDesc;
				memset(&resDesc, 0, sizeof(resDesc));
				resDesc.resType = cudaResourceTypeArray;
				resDesc.res.array.array = data;
				cudaTextureDesc texDesc;
				memset(&texDesc, 0, sizeof(texDesc));
				texDesc.addressMode[0] = _am;
				texDesc.filterMode = _fm;
				texDesc.readMode = _rm;
				texDesc.normalizedCoords = normalizedCoords;
				cudaCreateTextureObject(&textureObj, &resDesc, &texDesc, nullptr);
			}
		}
		Texture(cudaChannelFormatDesc const& channelDesc, cudaTextureAddressMode addressMode,
			cudaTextureFilterMode filterMode, cudaTextureReadMode readMode, bool normalizedCoords,
			void const* src, size_t width)
			:
			data(nullptr),
			textureObj(0)
		{
			if (width)
			{
				cudaMallocArray(&data, &channelDesc, 256);
				if (src)cudaMemcpyToArray(data, 0, 0, src,
					width * (channelDesc.x + channelDesc.y + channelDesc.z + channelDesc.w) / 8,
					cudaMemcpyHostToDevice);//only for dim<3
				cudaResourceDesc resDesc;
				memset(&resDesc, 0, sizeof(resDesc));
				resDesc.resType = cudaResourceTypeArray;
				resDesc.res.array.array = data;
				cudaTextureDesc texDesc;
				memset(&texDesc, 0, sizeof(texDesc));
				texDesc.addressMode[0] = addressMode;
				texDesc.filterMode = filterMode;
				texDesc.readMode = readMode;
				texDesc.normalizedCoords = normalizedCoords;
				cudaCreateTextureObject(&textureObj, &resDesc, &texDesc, nullptr);
			}
		}
		~Texture()
		{
			if (textureObj)
			{
				cudaDestroyTextureObject(textureObj);
				textureObj = 0;
			}
			if (data)
			{
				cudaFreeArray(data);
				data = nullptr;
			}
		}
		operator cudaTextureObject_t()const
		{
			return textureObj;
		}
	};
	template<>struct Texture<2>
	{
	};
	template<>struct Texture<3>
	{
	};
	struct TextureCube
	{
		cudaArray* data;
		cudaTextureObject_t textureObj;

		TextureCube(cudaChannelFormatDesc const& _cd, cudaTextureFilterMode _fm,
			cudaTextureReadMode _rm, bool normalizedCoords, CubeMap const& cubeMap)
			:
			data(nullptr),
			textureObj(0)
		{
			if (cubeMap.width)
			{
				cudaExtent extent{ cubeMap.width, cubeMap.width, 6 };
				cudaMalloc3DArray(&data, &_cd, extent, cudaArrayCubemap);
				cubeMap.moveToGPU(data);
				cudaResourceDesc resDesc;
				cudaTextureDesc texDesc;
				memset(&resDesc, 0, sizeof(resDesc));
				memset(&texDesc, 0, sizeof(texDesc));
				resDesc.resType = cudaResourceTypeArray;
				resDesc.res.array.array = data;
				texDesc.normalizedCoords = normalizedCoords;
				texDesc.filterMode = _fm;
				texDesc.addressMode[0] = cudaAddressModeWrap;
				texDesc.addressMode[1] = cudaAddressModeWrap;
				texDesc.addressMode[2] = cudaAddressModeWrap;
				texDesc.readMode = _rm;
				cudaCreateTextureObject(&textureObj, &resDesc, &texDesc, nullptr);
			}
		}
		~TextureCube()
		{
			if (textureObj)
			{
				cudaDestroyTextureObject(textureObj);
				textureObj = 0;
			}
			if (data)
			{
				cudaFreeArray(data);
				data = nullptr;
			}
		}
		operator cudaTextureObject_t()const
		{
			return textureObj;
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