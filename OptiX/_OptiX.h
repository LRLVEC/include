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
			::OpenGL::Buffer trianglesBuffer;
			::OpenGL::Buffer pixelBuffer;
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
					n += sweepStr(PTXLists.data + n, "%*[\r\n{]%n");
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
			RTbuffer buffer;
			unsigned int type;
			RTformat format;

			Buffer() = delete;
			Buffer(bool)
				:
				context(0),
				buffer(0),
				type(RT_BUFFER_INPUT),
				format(RT_FORMAT_FLOAT)
			{
			}
			Buffer(RTcontext* _context, unsigned int _type, RTformat _format)
				:
				context(_context),
				type(_type),
				format(_format)
			{
				create();
			}
			Buffer(RTcontext* _context, unsigned int _type, RTformat _format, GLuint _pbo)
				:
				context(_context),
				type(_type),
				format(_format)
			{
				create(_pbo);
			}
			Buffer(Buffer const& a)
				:
				context(a.context),
				buffer(a.buffer),
				type(a.type),
				format(a.format)
			{
			}
			operator RTbuffer()
			{
				return buffer;
			}
			operator RTbuffer* ()
			{
				return &buffer;
			}
			operator RTobject* ()
			{
				return (RTobject*)& buffer;
			}
			void create()
			{
				rtBufferCreate(*context, type, &buffer);
				rtBufferSetFormat(buffer, format);
			}
			void create(GLuint _pbo)
			{
				rtBufferCreateFromGLBO(*context, type, _pbo, &buffer);
				rtBufferSetFormat(buffer, format);
			}
			void destory()
			{
				rtBufferDestroy(buffer);
			}
			void reg()
			{
				rtBufferGLRegister(buffer);
			}
			void unreg()
			{
				rtBufferGLUnregister(buffer);
			}
			void* map()
			{
				void* t;
				rtBufferMap(buffer, &t);
				return t;
			}
			void unmap()
			{
				rtBufferUnmap(buffer);
			}
			unsigned long long getSize1()const
			{
				unsigned long long a(0);
				rtBufferGetSize1D(buffer, &a);
				return a;
			}
			Math::vec2<unsigned long long> getSize2()const
			{
				Math::vec2<unsigned long long> a(0);
				rtBufferGetSize2D(buffer, &a[0], &a[1]);
				return a;
			}
			Math::vec3<unsigned long long> getSize3()const
			{
				Math::vec3<unsigned long long> a(0);
				rtBufferGetSize3D(buffer, &a[0], &a[1], &a[2]);
				return a;
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
				rtBufferSetSize3D(buffer, _width, _height, _depth);
			}
			void readBMP(BMP const& bmp)
			{
				setSize(bmp.header.width, bmp.header.height);
				unsigned int* p((unsigned int*)map());
				for (int c0(0); c0 < bmp.header.height; ++c0)
					for (int c1(0); c1 < bmp.header.width; ++c1)
						::memcpy(p + bmp.header.width * c0 + c1, bmp.data + bmp.header.width * c0 + c1, 3);
				unmap();
			}
			void readCube(BMPCube const& cube)
			{
				unsigned long long width(cube.bmp[0].header.width);
				unsigned long long height(cube.bmp[0].header.height);
				unsigned long long wh(width * height);
				setSize(width, height, 6);
				unsigned int* p((unsigned int*)map());
				for (unsigned long long c0(0); c0 < 6; ++c0)
					for (unsigned long long c1(0); c1 < height; ++c1)
						for (unsigned long long c2(0); c2 < width; ++c2)
							p[wh * c0 + width * c1 + c2] =
							cube.bmp[c0].data[width * c1 + c2].rgb();
				unmap();
			}
		};
		struct VariableBase
		{
			RTvariable variable;
			String<char> name;
			struct Data
			{
				static bool constexpr valid = true;
				virtual void* pointer() = 0;
				virtual unsigned long long size() = 0;
			};
			VariableBase() = delete;
			VariableBase(String<char>const& _name)
				:
				variable(0),
				name(_name)
			{
			}
			VariableBase(VariableBase const& a)
				:
				variable(a.variable),
				name(a.name)
			{
			}
			void setObject(RTobject* _object)
			{
				rtVariableSetObject(variable, *_object);
			}
			void set1f(float x)
			{
				rtVariableSet1f(variable, x);
			}
			void set2f(float x, float y)
			{
				rtVariableSet2f(variable, x, y);
			}
			void set3f(float x, float y, float z)
			{
				rtVariableSet3f(variable, x, y, z);
			}
			void set4f(float x, float y, float z, float w)
			{
				rtVariableSet4f(variable, x, y, z, w);
			}
			void set1u(unsigned int x)
			{
				rtVariableSet1ui(variable, x);
			}
			void setUser(const void* _data, unsigned long long _size)
			{
				rtVariableSetUserData(variable, _size, _data);
			}
		};
		template<class T>struct Variable :VariableBase
		{
			static_assert(T::valid, "Not a valid VariableBase::Data type!");
			RTcontext* context;
			T* data;
			Variable(RTcontext* _context, String<char> const& _name, T* _data)
				:
				VariableBase(_name),
				context(_context),
				data(_data)
			{
				declare();
			}
			void declare()
			{
				rtContextDeclareVariable(*context, name, &variable);
			}
			void set()
			{
				rtVariableSetUserData(variable, data->size(), data->pointer());
			}
		};
		template<>struct Variable<RTcontext> :VariableBase
		{
			RTcontext* context;

			Variable(RTcontext* _context, String<char> const& _name)
				:
				VariableBase(_name),
				context(_context)
			{
				declare();
			}
			void declare()
			{
				rtContextDeclareVariable(*context, name, &variable);
			}
		};
		template<>struct Variable<RTprogram> :VariableBase
		{
			RTprogram* program;

			Variable(RTprogram* _program, String<char> const& _name)
				:
				VariableBase(_name),
				program(_program)
			{
				declare();
			}
			void declare()
			{
				rtProgramDeclareVariable(*program, name, &variable);
			}
		};
		template<>struct Variable<RTmaterial> :VariableBase
		{
			RTmaterial* material;

			Variable(RTmaterial* _material, String<char> const& _name)
				:
				VariableBase(_name),
				material(_material)
			{
				declare();
			}
			void declare()
			{
				rtMaterialDeclareVariable(*material, name, &variable);
			}
		};
		template<>struct Variable<RTgeometrytriangles> :VariableBase
		{
			RTgeometrytriangles* geometryTriangles;

			Variable(RTgeometrytriangles* geoTriangles, String<char> const& _name)
				:
				VariableBase(_name),
				geometryTriangles(geoTriangles)
			{
				declare();
			}
			void declare()
			{
				rtGeometryTrianglesDeclareVariable(*geometryTriangles, name, &variable);
			}
		};
		struct Program
		{
			RTcontext* context;
			RTprogram program;
			String<char>* source;
			String<char>name;

			Program() = delete;
			Program(bool)
				:
				context(nullptr),
				program(nullptr),
				source(nullptr)
			{
			}
			Program(RTcontext* _context, PTXManager const& _pm, String<char>_name)
				:
				context(_context),
				source(&_pm.getFunction(_name)),
				name(_name)
			{
				create();
			}
			operator RTprogram()const
			{
				return program;
			}
			operator RTprogram* ()
			{
				return &program;
			}
			void create()
			{
				rtProgramCreateFromPTXString(*context, *source, name, &program);
			}
			void destory()
			{
				rtProgramDestroy(program);
			}
		};
		struct Acceleration
		{
			enum Builder
			{
				Trbvh,
				Sbvh,
				Bvh,
				NoAccel
			};
			RTcontext* context;
			RTacceleration acceleration;
			Builder builder;
			Acceleration(RTcontext* _context, Builder _builder)
				:
				context(_context)
			{
				rtAccelerationCreate(*context, &acceleration);
				setBuilder(_builder);
			}
			operator RTacceleration()const
			{
				return acceleration;
			}
			operator RTacceleration* ()
			{
				return &acceleration;
			}
			void destory()
			{
				rtAccelerationDestroy(acceleration);
			}
			void setBuilder(Builder _builder)
			{
				switch (builder = _builder)
				{
					case Trbvh:rtAccelerationSetBuilder(acceleration, "Trnbvh"); break;
					case Sbvh:rtAccelerationSetBuilder(acceleration, "Sbvh"); break;
					case Bvh:rtAccelerationSetBuilder(acceleration, "Bvh"); break;
					case NoAccel:rtAccelerationSetBuilder(acceleration, "NoAccel"); break;
				}
			}
		};
		struct Material
		{
			struct AnyHit
			{
				RTmaterial* material;
				RTprogram* program;
				unsigned int rayType;
				AnyHit() = delete;
				AnyHit(RTmaterial* _material, RTprogram* _program, unsigned int _rayType)
					:
					material(_material),
					program(_program),
					rayType(_rayType)
				{
				}
				void setProgram()
				{
					rtMaterialSetAnyHitProgram(*material, rayType, *program);
				}
			};
			struct CloseHit
			{
				RTmaterial* material;
				RTprogram* program;
				unsigned int rayType;
				CloseHit() = delete;
				CloseHit(RTmaterial* _material, RTprogram* _program, unsigned int _rayType)
					:
					material(_material),
					program(_program),
					rayType(_rayType)
				{
				}
				void setProgram()
				{
					rtMaterialSetClosestHitProgram(*material, rayType, *program);
				}
			};

			RTmaterial material;
			Material() = delete;
			Material(Material const&) = delete;
			Material(RTcontext* _context)
			{
				rtMaterialCreate(*_context, &material);
			}
			operator RTmaterial()const
			{
				return material;
			}
			operator RTmaterial* ()
			{
				return &material;
			}
			void destory()
			{
				rtMaterialDestroy(material);
			}
		};
		struct Geometry
		{
			RTcontext* context;
			RTgeometry geometry;
			unsigned int primitiveCount;
			Geometry(RTcontext* _context)
				:
				context(_context),
				primitiveCount(0)
			{
				rtGeometryCreate(*context, &geometry);
			}
			void setPrivitiveCount(unsigned int _primitiveCount)
			{
				primitiveCount = _primitiveCount;
				rtGeometrySetPrimitiveCount(geometry, _primitiveCount);
			}
		};
		struct GeometryTriangles
		{
			struct TrianglesBuffers
			{
				struct Parameters
				{
					String<char> name;
					RTbuffertype type;
					RTformat format;
				};
				struct BufferVariable
				{
					Buffer buffer;
					Variable<RTgeometrytriangles> variable;
					BufferVariable(Parameters const& _para, RTcontext* _context, RTgeometrytriangles* geoTriangles)
						:
						buffer(_context, _para.type, _para.format),
						variable(geoTriangles, _para.name)
					{
						buffer.setSize(0);
						variable.setObject(buffer);
					}
					bool operator==(String<char>const& _name)
					{
						return variable.name == _name;
					}
					void destory()
					{
						buffer.destory();
					}
				};
				Vector<BufferVariable>buffers;
				TrianglesBuffers(RTcontext* _context, RTgeometrytriangles* geoTriangles, Vector<Parameters>const& paras)
				{
					for (int c0(0); c0 < paras.length; ++c0)
						buffers.pushBack(BufferVariable(paras.data[c0], _context, geoTriangles));
				}
				BufferVariable& operator[](String<char>const& name)
				{
					return buffers.findFirst(name);
				}
				void destory()
				{
					buffers.traverse([](BufferVariable& a) { a.destory(); return true; });
				}
			};
			RTcontext* context;
			RTgeometrytriangles triangles;
			Program attrib;
			unsigned long long count;
			unsigned int materialNum;
			RTgeometrybuildflags buildFlag;
			TrianglesBuffers buffers;
			GeometryTriangles(RTcontext* _context, unsigned int _materialCount, RTgeometrybuildflags _buildFlag,
				Vector<TrianglesBuffers::Parameters>const& paras)
				:
				context((rtGeometryTrianglesCreate(*_context, &triangles), _context)),
				count(0),
				materialNum(_materialCount),
				buildFlag(_buildFlag),
				buffers(_context, &triangles, paras),
				attrib(false)
			{
				rtGeometryTrianglesSetMaterialCount(triangles, _materialCount);
				rtGeometryTrianglesSetBuildFlags(triangles, _buildFlag);
			}
			GeometryTriangles(RTcontext* _context, PTXManager const& _pm, String<char>const& _attribProgram,
				unsigned int _materialCount, RTgeometrybuildflags _buildFlag,
				Vector<TrianglesBuffers::Parameters>const& paras)
				:
				context((rtGeometryTrianglesCreate(*_context, &triangles), _context)),
				count(0),
				materialNum(_materialCount),
				buildFlag(_buildFlag),
				buffers(_context, &triangles, paras),
				attrib(_context, _pm, _attribProgram)
			{
				rtGeometryTrianglesSetMaterialCount(triangles, _materialCount);
				rtGeometryTrianglesSetBuildFlags(triangles, _buildFlag);
				setAttribProgram();
			}
			operator RTgeometrytriangles()const
			{
				return triangles;
			}
			operator RTgeometrytriangles* ()
			{
				return &triangles;
			}
			void setCount(unsigned long long _count)
			{
				rtGeometryTrianglesSetPrimitiveCount(triangles, count = _count);
			}
			void setVertices(Buffer* a, unsigned int vertexCount, RTsize offset, RTsize stride)
			{
				rtGeometryTrianglesSetVertices(triangles, vertexCount, a->buffer, offset, stride, a->format);
			}
			void setVertices(String<char>const& name, unsigned int vertexCount, RTsize offset, RTsize stride)
			{
				Buffer* a(&buffers[name].buffer);
				rtGeometryTrianglesSetVertices(triangles, vertexCount, a->buffer, offset, stride, a->format);
			}
			void setIndices(Buffer* a, RTsize offset, RTsize stride)
			{
				rtGeometryTrianglesSetTriangleIndices(triangles, a->buffer, offset, stride, a->format);
			}
			void setIndices(String<char>const& name, RTsize offset, RTsize stride)
			{
				Buffer* a(&buffers[name].buffer);
				rtGeometryTrianglesSetTriangleIndices(triangles, a->buffer, offset, stride, a->format);
			}
			void setAttribProgram()
			{
				rtGeometryTrianglesSetAttributeProgram(triangles, attrib);
			}
			void addSTL(String<char>const& name, STL const&, unsigned int num);
			void addSTL(String<char>const& vertices, String<char>const& normals, String<char>const& indices, STL const&);
			void destory()
			{
				rtGeometryTrianglesDestroy(triangles);
				buffers.destory();
			}
		};
		struct GeometryInstance
		{
			RTcontext* context;
			RTgeometryinstance instance;
			unsigned int materialCount;
			GeometryInstance(RTcontext* _context)
				:
				context(_context),
				materialCount(0)
			{
				rtGeometryInstanceCreate(*_context, &instance);
			}
			operator RTgeometryinstance()const
			{
				return instance;
			}
			operator RTgeometryinstance* ()
			{
				return &instance;
			}
			void setTriangles(GeometryTriangles const& triangles)
			{
				rtGeometryInstanceSetGeometryTriangles(instance, triangles);
			}
			void setMaterial(Vector<Material*>const& materials)
			{
				rtGeometryInstanceSetMaterialCount(instance, materialCount = materials.length);
				for (int c0(0); c0 < materials.length; ++c0)
					rtGeometryInstanceSetMaterial(instance, c0, materials.data[c0]->material);
			}
			void destory()
			{
				rtGeometryInstanceDestroy(instance);
			}
		};
		struct GeometryGroup
		{
			RTcontext* context;
			RTgeometrygroup geoGroup;
			Acceleration accel;
			unsigned int childCount;
			GeometryGroup(RTcontext* _context, Acceleration::Builder _accelBuilder)
				:
				context(_context),
				accel(_context, _accelBuilder),
				childCount(0)
			{
				rtGeometryGroupCreate(*context, &geoGroup);
				setAccel();
			}
			operator RTgeometrygroup()const
			{
				return geoGroup;
			}
			operator RTgeometrygroup* ()
			{
				return &geoGroup;
			}
			operator RTobject* ()
			{
				return (RTobject*)& geoGroup;
			}
			void destory()
			{
				accel.destory();
				rtGeometryGroupDestroy(geoGroup);
			}
			void setInstance(Vector<GeometryInstance*>const& instances)
			{
				rtGeometryGroupSetChildCount(geoGroup, childCount = instances.length);
				for (int c0(0); c0 < instances.length; ++c0)
					rtGeometryGroupSetChild(geoGroup, c0, instances.data[c0]->instance);
			}
			void setAccel()
			{
				rtGeometryGroupSetAcceleration(geoGroup, accel);
			}
		};
		struct Transform
		{
			RTcontext* context;
			RTtransform trans;
			Math::mat4 <float>mat;
			//row major!!
			Transform(RTcontext* _context)
				:
				context(_context),
				mat(Math::mat4<float>::id())
			{
				rtTransformCreate(*context, &trans);
				rtTransformSetMatrix(trans, 0, mat.array[0], 0);
			}
			Transform(RTcontext* _context, Math::mat4<float>const& _mat)
				:
				context(_context),
				mat(_mat)
			{
				rtTransformCreate(*context, &trans);
				rtTransformSetMatrix(trans, 0, mat.array[0], 0);
			}
			operator RTtransform()const
			{
				return trans;
			}
			operator RTtransform* ()
			{
				return &trans;
			}
			operator RTobject* ()
			{
				return (RTobject*)& trans;
			}
			void setMat(Math::mat4<float>const& _mat)
			{
				rtTransformSetMatrix(trans, 0, (mat = _mat).array[0], 0);
			}
			void setChild(RTobject* object)
			{
				rtTransformSetChild(trans, *object);
			}
		};
		struct Group
		{
			RTcontext* context;
			RTgroup group;
			Acceleration accel;
			Variable<RTcontext> groupGPU;
			unsigned int childCount;
			Group(RTcontext* _context, String<char>const& _name, Acceleration::Builder _accelBuilder)
				:
				context(_context),
				accel(_context, _accelBuilder),
				groupGPU(context, _name),
				childCount(0)
			{
				rtGroupCreate(*context, &group);
				setAccel();
				setGroupGPU();
			}
			operator RTgroup()const
			{
				return group;
			}
			operator RTgroup* ()
			{
				return &group;
			}
			operator RTobject* ()
			{
				return (RTobject*)& group;
			}
			void destory()
			{
				accel.destory();
				rtGroupDestroy(group);
			}
			void setGeoGroup(Vector<RTobject*>const& geoGroup)
			{
				rtGroupSetChildCount(group, childCount = geoGroup.length);
				for (int c0(0); c0 < geoGroup.length; ++c0)
					rtGroupSetChild(group, c0, *geoGroup.data[c0]);
			}
			void setAccel()
			{
				rtGroupSetAcceleration(group, accel);
			}
			void setGroupGPU()
			{
				groupGPU.setObject(*this);
			}
		};
		struct Context
		{
			struct Miss
			{
				unsigned int rayType;
				Program miss;
				Miss(RTcontext* _context, PTXManager const& _pm, String<char>_name, unsigned int _rayType)
					:
					miss(_context, _pm, _name),
					rayType(_rayType)
				{
				}
				void setMissProgram()
				{
					rtContextSetMissProgram(*miss.context, rayType, miss);
				}
			};
			struct EntryPoint
			{
				unsigned int entryPoint;
				Program entry;
				EntryPoint(RTcontext* _context, PTXManager const& _pm, String<char>_name, unsigned int _entryPoint)
					:
					entry(_context, _pm, _name),
					entryPoint(_entryPoint)
				{
				}
				void setEntryPoint()
				{
					rtContextSetRayGenerationProgram(*entry.context, entryPoint, entry);
				}
			};

			RTcontext context;
			Vector<EntryPoint>entries;
			Vector<Miss>misses;
			unsigned int rayTypeCount;
			unsigned int maxDepth;

			Context() = delete;
			Context(PTXManager const& _pm, Vector<Pair<unsigned int, String<char>>>const& _entries,
				Vector<Pair<unsigned int, String<char>>>const& _misses,
				unsigned int _rayTypeCount, unsigned int _maxDepth)
				:
				context(0),
				rayTypeCount(_rayTypeCount),
				maxDepth(_maxDepth)
			{
				create();
				rtContextSetEntryPointCount(context, _entries.length);
				for (int c0(0); c0 < _entries.length; ++c0)
				{
					entries.pushBack(EntryPoint(&context, _pm, _entries.data[c0].data1, _entries.data[c0].data0));
					entries.end().setEntryPoint();
				}
				for (int c0(0); c0 < _misses.length; ++c0)
				{
					misses.pushBack(Miss(&context, _pm, _misses.data[c0].data1, _misses.data[c0].data0));
					misses.end().setMissProgram();
				}
			}
			operator RTcontext()const
			{
				return context;
			}
			operator RTcontext* ()
			{
				return &context;
			}
			void create()
			{
				int a(1);
				rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(int), &a);
				rtContextCreate(&context);
				rtContextSetRayTypeCount(context, rayTypeCount);
				rtContextSetMaxTraceDepth(context, maxDepth);
			}
			void destory()
			{
				rtContextDestroy(context);
			}
			void printAllDeviceInfo()const
			{
				unsigned int n;
				rtDeviceGetDeviceCount(&n);
				::printf("Devices count: %u\n", n);
				for (int i = 0; i < n; ++i)
				{
					char name[256];
					char pciBusId[16];
					int computeCaps[2];
					RTsize total_mem;
					int clock_rate;
					int threads_per_block;
					int sm_count;
					int execution_timeout_enabled;
					int texture_count;
					int tcc_driver;
					int cuda_device_ordinal;

					rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_NAME, sizeof(name), name);
					rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_PCI_BUS_ID, sizeof(pciBusId), pciBusId);
					printf("Device %d (%s): %s\n", i, pciBusId, name);

					rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(computeCaps), &computeCaps);
					printf("  Compute Support: %d %d\n", computeCaps[0], computeCaps[1]);

					rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(total_mem), &total_mem);
					printf("  Total Memory: %llu bytes\n", (unsigned long long)total_mem);

					rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_CLOCK_RATE, sizeof(clock_rate), &clock_rate);
					printf("  Clock Rate: %u kilohertz\n", clock_rate);

					rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, sizeof(threads_per_block), &threads_per_block);
					printf("  Max. Threads per Block: %u\n", threads_per_block);

					rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, sizeof(sm_count), &sm_count);
					printf("  SM Count: %u\n", sm_count);

					rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED, sizeof(execution_timeout_enabled), &execution_timeout_enabled);
					printf("  Execution Timeout Enabled: %d\n", execution_timeout_enabled);

					rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_MAX_HARDWARE_TEXTURE_COUNT, sizeof(texture_count), &texture_count);
					printf("  Max. HW Texture Count: %u\n", texture_count);

					rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_TCC_DRIVER, sizeof(tcc_driver), &tcc_driver);
					printf("  TCC driver enabled: %u\n", tcc_driver);

					rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(cuda_device_ordinal), &cuda_device_ordinal);
					printf("  CUDA Device Ordinal: %d\n", cuda_device_ordinal);

					Vector<int> compatible_devices;
					compatible_devices.malloc(n + 1);
					rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_COMPATIBLE_DEVICES, sizeof(compatible_devices), &compatible_devices[0]);
					printf("  Compatible devices: ");
					if (compatible_devices[0] == 0)printf("none\n");
					else
					{
						for (int i = 0; i < compatible_devices[0]; ++i)
						{
							if (i > 0)printf(", ");
							printf("%d", compatible_devices[i + 1]);
						}
						printf("\n");
					}
					printf("\n");
				}
			}
			void printDeviceInfo()const
			{
				unsigned int n(0);
				rtContextGetDeviceCount(context, &n);
				::printf("Devices count:%d\n", n);
				int* d((int*)::malloc(sizeof(int) * 3));
				rtDeviceGetAttribute(0, RT_DEVICE_ATTRIBUTE_COMPATIBLE_DEVICES, sizeof(int) * (n + 1), d);
				::printf("Compatible devices count:%d\n", d[0]);
				::printf("Compatible device(s): ");
				for (int c0(0); c0 < d[0]; ++c0)
					::printf("%d ", d[c0 + 1]);
				::printf("\n");
			}
			void pringStackSize()const
			{
				unsigned long long a(0);
				rtContextGetStackSize(context, (RTsize*)& a);
				::printf("Stack size: %u\n", a);
			}
			void setStackSize(unsigned int size)
			{
				rtContextSetStackSize(context, size);
			}
			/*void setDevice(int device)
			{
				rtContextSetDevices(context, 1, &device);
			}*/
			void validate()
			{
				rtContextValidate(context);
			}
			void launch(unsigned int _entryPoint, unsigned int x)
			{
				rtContextLaunch1D(context, _entryPoint, x);
			}
			void launch(unsigned int _entryPoint, unsigned int x, unsigned int y)
			{
				rtContextLaunch2D(context, _entryPoint, x, y);
			}
			void launch(unsigned int _entryPoint, unsigned int x, unsigned int y, unsigned int z)
			{
				rtContextLaunch3D(context, _entryPoint, x, y, z);
			}
		};
		struct Trans
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

				RTcontext* context;
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
					y(_persp.y),
					updated(false)
				{
				}
				Perspective(Perspective const&) = default;
				void init(::OpenGL::FrameScale const& _size)
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
			struct BufferData : VariableBase::Data
			{
				struct Trans
				{
					Math::mat<float, 3, 4>ans;
					Math::vec3<float>r0;
					float z0;
				};
				Trans trans;
				BufferData()
				{
				}
				virtual void* pointer()override
				{
					return &trans;
				}
				virtual unsigned long long size()override
				{
					return sizeof(Trans);
				}
			};

			RTcontext* context;
			Perspective persp;
			Scroll scroll;
			Key key;
			Mouse mouse;
			BufferData bufferData;
			Variable<BufferData>variable;
			Math::vec3<double>dr;
			Math::mat3<double>trans;
			double depth;
			bool moved;
			bool updated;

			Trans() = delete;
			Trans(Data const& _data)
				:
				context(_data.context),
				persp(_data.persp),
				scroll(_data.scroll),
				key(_data.key),
				mouse(),
				moved(true),
				updated(false),
				bufferData(),
				variable(_data.context, "trans", &bufferData),
				dr(_data.initialPosition),
				trans(Math::mat3<double>::id()),
				depth(_data.depth)
			{
			}
			void init(::OpenGL::FrameScale const& _size)
			{
				persp.init(_size);
				bufferData.trans.z0 = -float(_size.h) / tan(Math::Pi * persp.fovy / 360.0);
				calcAns();
				updated = true;
				variable.set();
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
					variable.set();
					updated = true;
				}
			}
		};
		struct TransDepth
		{
			struct Data
			{
				struct Perspective
				{
					double fovy;
					double D;
					double V;
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

				RTcontext* context;
				Perspective persp;
				Scroll scroll;
				Key key;
				Math::vec3<double> initialPosition;
				double depth;
			};
			struct Perspective
			{
				double fovy;
				unsigned int F;
				double D;
				double V;
				bool updated;
				Perspective() = delete;
				Perspective(Data::Perspective const& _persp)
					:
					fovy(_persp.fovy),
					D(_persp.D),
					V(_persp.V),
					updated(false)
				{
				}
				Perspective(Perspective const&) = default;
				void init(::OpenGL::FrameScale const& _size)
				{
					F = sqrt(_size.w * _size.w + _size.h * _size.h) / 2.0;
					updated = true;
				}
				void increaseV(double dv)
				{
					V = V + dv > 0 ? V + dv : V;
					updated = true;
				}
				void increaseD(double dd)
				{
					D = D + dd > 0 ? D + dd : D;
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
			struct BufferData : VariableBase::Data
			{
				struct Trans
				{
					Math::mat<float, 3, 4>ans;
					Math::vec3<float>r0;
					float V;
					float P;
					float D;
				};
				Trans trans;
				BufferData()
				{
				}
				void setLense(Perspective const& _persp)
				{
					trans.P = _persp.V * tan(Math::Pi * _persp.fovy / 360.0) / _persp.F;
					trans.D = _persp.D;
					trans.V = -_persp.V;
				}
				virtual void* pointer()override
				{
					return &trans;
				}
				virtual unsigned long long size()override
				{
					return sizeof(Trans);
				}
			};

			RTcontext* context;
			Perspective persp;
			Scroll scroll;
			Key key;
			Mouse mouse;
			BufferData bufferData;
			Variable<BufferData>variable;
			Math::vec3<double>dr;
			Math::mat3<double>trans;
			double depth;
			bool moved;
			bool updated;

			TransDepth() = delete;
			TransDepth(Data const& _data)
				:
				context(_data.context),
				persp(_data.persp),
				scroll(_data.scroll),
				key(_data.key),
				mouse(),
				moved(true),
				updated(false),
				bufferData(),
				variable(_data.context, "transDepth", &bufferData),
				dr(_data.initialPosition),
				trans(Math::mat3<double>::id()),
				depth(_data.depth)
			{
			}
			void init(::OpenGL::FrameScale const& _size)
			{
				persp.init(_size);
				persp.updated = false;
				bufferData.setLense(persp);
				calcAns();
				updated = true;
				variable.set();
			}
			void resize(::OpenGL::FrameScale const& _size)
			{
				persp.init(_size);
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
					bufferData.setLense(persp);
					persp.updated = false;
					operated = true;
				}
				if (operated)
				{
					calcAns();
					variable.set();
					updated = true;
				}
			}
		};
		struct RayTracer
		{
			virtual void run() = 0;
			virtual void resize(FrameScale const& _size) = 0;
			virtual void terminate() = 0;
		};
	}
}