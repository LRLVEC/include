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
		struct DefautRenderer :Program
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
					return sizeof(Triangle)* triangles.length;
				}
			};
			struct PixelData :Buffer::Data
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
					return frameConfig.width* frameConfig.height * 16;
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
					return name == _name;
				}
			};
			File* folder;
			Vector<Source>sources;

			PTXManager(File* _folder)
				:
				folder(_folder)
			{
				getSource();
			}
			void getSource(Vector<String<char>>const& _names)
			{
				File& ptx(folder->find("ptx"));
				String<char>PTXLists(ptx.find("PTXLists.txt").readText());
				Vector<int>functions(PTXLists.find("Program:"));
				for (int c0(0); c0 < functions.length; ++c0)
				{
					int n(functions[c0]);
					int delta;
					char t[100];
					n += sweepStr(PTXLists.data + n, "Program:%n");
					sscanf(PTXLists.data + n, "%[a-zA-Z0-9_]%n", t, &delta);
					n += delta;
					Source s;
					s.name = t;
					s.source = ptx.findInThis(s.name + ".cu").readText();

				}
				for (int c0(0); c0 < _names.length; ++c0)
				{
					String<char>s(ptx.findInThis(_names.data[c0] + ".ptx").readText());
					if (s.length)sources.pushBack({ _names.data[c0],s });
				}
			}
			String<char>const& getFunction(String<char>_name)const
			{
				for (int c0(0); c0 < sources.length; ++c0)
					if (sources.data[c0] == _name)
						return sources.data[c0].source;
			}
		};
		struct Function
		{
			RTcontext* context;
			RTprogram program;
			String<char>source;
			String<char>name;

			Function() = delete;
			Function(RTcontext* _context, String<char>const& _source, String<char>_name)
				:
				context(_context),
				source(_source),
				name(_name)
			{
				create();
			}
			void create()
			{
				rtProgramCreateFromPTXString(*context, source, name, &program);
			}
			operator RTprogram()const
			{
				return program;
			}
		};

		struct RayTracer
		{
			RTcontext context;
			Function* rayGen;
			RTbuffer  outBuffer;

		};

	}
}