#pragma once
#include <GL/_OpenGL.h>
#include <GL/_Texture.h>
#include <_Math.h>
#include <_Pair.h>
#include <_Array.h>
#include <optix.h>
#include <optix_gl_interop.h>

namespace OpenGL
{
	namespace OptiX
	{
		struct DefautRenderer : ::OpenGL::Program
		{
			struct TriangleData :Buffer::Data
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
					Data(DynamicDraw),
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
			bool updated;

			DefautRenderer(SourceManager* _sourceManage, FrameScale const& _size)
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
			void resize(int _w, int _h)
			{
				glViewport(0, 0, _w, _h);
				pixelData.frameConfig.resize(_w, _h);
				pixelPixel.dataInit();
			}
			operator GLuint ()const
			{
				return pixelBuffer.buffer;
			}
		};
		struct PTXManager
		{
			struct Source
			{
				String<char>name;
				String<char>source;
				Vector<String<char>>functions;
				bool operator==(String<char> const& _name)const
				{
					for (int c0(0); c0 < functions.length; ++c0)
						if (_name == functions.data[c0])
							return true;
				}
			};
			File* folder;
			Vector<Source>sources;

			PTXManager(File* _folder)
				:
				folder(_folder)
			{
				readSource();
			}
			void readSource()
			{
				File& ptx(folder->find("ptx"));
				String<char>PTXLists(ptx.find("PTXLists.txt").readText());
				Vector<int>files(PTXLists.find("Program:"));
				for (int c0(0); c0 < files.length; ++c0)
				{
					int n(files[c0]);
					int delta;
					char t[100];
					char t1[10];
					n += sweepStr(PTXLists.data + n, "Program:%*[^a-zA-Z0-9_]%n");
					sscanf(PTXLists.data + n, "%[a-zA-Z0-9_]%n", t, &delta);
					n += delta;
					Source s;
					s.name = t;
					s.source = ptx.findInThis(s.name + ".ptx").readText();
					n += sweepStr(PTXLists.data + n, "%*[^{]%{%n");
					int j(0);
					do
					{
						j = sscanf(PTXLists.data + n, "%s%n%*[\t\r\n ]%[}]", t, &delta, t1);
						n += delta;
						if (j < 1)break;
						s.functions.pushBack(t);
					}
					while (j == 1);
					sources.pushBack(s);
				}
			}
			String<char>& getFunction(String<char>const& _name)const
			{
				for (int c0(0); c0 < sources.length; ++c0)
					if (sources.data[c0] == _name)
						return sources.data[c0].source;
			}
		};
		struct Buffer
		{
			RTcontext* context;
			RTbuffer  buffer;
			RTbuffertype type;
			RTformat format;

			Buffer(RTcontext* _context, RTbuffertype _type, RTformat _format, GLuint _pbo)
				:
				context(_context),
				type(_type),
				format(_format)
			{
				create(_pbo);
			}
			void create(GLuint _pbo)
			{
				rtBufferCreateFromGLBO(*context, type, _pbo, &buffer);
			}
			void setFormat()
			{
				rtBufferSetFormat(buffer, format);
			}
			void setElementSize(unsigned int _size)
			{
				if (type == RT_FORMAT_USER)
					rtBufferSetElementSize(buffer, _size);
			}
			void setSize(unsigned int _width)
			{
				rtBufferSetSize1D(buffer, _width);
			}
			void setSize(unsigned int _width, unsigned int _height)
			{
				rtBufferSetSize2D(buffer, _width, _height);
			}
			void setSize(unsigned int _width, unsigned int _height, unsigned int _depth)
			{
				rtBufferSetSize3D(buffer, _width, _height, _height);
			}

		};
		struct Variable
		{
			RTvariable variable;

		};
		struct Program
		{
			RTcontext* context;
			RTprogram program;
			String<char>* source;
			String<char>name;

			Program() = delete;
			Program(RTcontext* _context, PTXManager const& _pm, String<char>_name)
				:
				context(_context),
				source(&_pm.getFunction(_name)),
				name(_name)
			{
				create();
			}
			void create()
			{
				rtProgramCreateFromPTXString(*context, *source, name, &program);
			}
			operator RTprogram()const
			{
				return program;
			}
		};
		struct Context
		{
			RTcontext context;
			Vector<Program*>entryPoints;
			unsigned int rayTypeCount;

			Context() = delete;
			Context(Vector<Program*>const& _entryPoints, unsigned int _rayTypeCount)
				:
				context(0),
				entryPoints(_entryPoints),
				rayTypeCount(_rayTypeCount)
			{
				create();
			}
			operator RTcontext()const
			{
				return context;
			}
			void create()
			{
				rtContextCreate(&context);
			}
			void init()
			{
				rtContextSetRayTypeCount(context, rayTypeCount);
				rtContextSetEntryPointCount(context, entryPoints.length);
				for (int c0(0); c0 < entryPoints.length; ++c0)
					rtContextSetRayGenerationProgram(context, c0, *entryPoints[c0]);
			}
			void validate()
			{
				rtContextValidate(context);
			}
			void launch(unsigned int _entryPoint, unsigned int x)
			{
				rtContextLaunch1D(context, _entryPoint, x);
			}
			void destory()
			{
				rtContextDestroy(context);
			}
		};

		struct RayTracer
		{
			RTcontext context;
			Program* rayGen;
			RTbuffer  outBuffer;

		};

	}
}