#pragma once
#define GLEW_STATIC
#include <GL/GLEW/glew.h>
#include <GL/GLFW/glfw3.h>
#pragma comment(lib,"OpenGL32.lib")
#ifdef _DEBUG
#pragma comment(lib,"GL/glew32s.lib")
#pragma comment(lib,"GL/glfw3dll.lib")
#else
#pragma comment(lib,"GL/glew32s.lib")
#pragma comment(lib,"GL/glfw3dll.lib")
#endif

#include <_Vector.h>
#include <_String.h>
#include <_Array.h>
#include <_Pair.h>
#include <_File.h>
#include <_Math.h>


namespace OpenGL
{
	struct FrameScale
	{
		int w;
		int h;
	};
	//GLFW GLEW initialization
	struct OpenGLInit
	{
		static bool initialized;
		static unsigned int versionMajor;
		static unsigned int versionMinor;

		OpenGLInit();
		OpenGLInit(unsigned int, unsigned int);

		void setOpenGLVersion(unsigned int, unsigned int);
	};
	//OpenGL class
	struct OpenGL
	{
		virtual void init(FrameScale const&) = 0;
		virtual void run() {}
		virtual void frameSize(int, int) = 0;
		virtual void framePos(int, int) = 0;
		virtual void frameFocus(int) = 0;
		virtual void mouseButton(int, int, int) = 0;
		virtual void mousePos(double, double) = 0;
		virtual void mouseScroll(double, double) = 0;
		virtual void key(GLFWwindow*, int, int, int, int) = 0;
	};
	enum ShaderType
	{
		VertexShader = GL_VERTEX_SHADER,
		TessControlShader = GL_TESS_CONTROL_SHADER,
		TessEvaluationShader = GL_TESS_EVALUATION_SHADER,
		GeometryShader = GL_GEOMETRY_SHADER,
		FragmentShader = GL_FRAGMENT_SHADER,
		ComputeShader = GL_COMPUTE_SHADER,
	};
	enum BufferType
	{
		None = 0,
		ArrayBuffer = GL_ARRAY_BUFFER,
		AtomicCounterBuffer = GL_ATOMIC_COUNTER_BUFFER,			//binding
		CopyReadBuffer = GL_COPY_READ_BUFFER,
		CopyWriteBuffer = GL_COPY_WRITE_BUFFER,
		DispatchIndirectBuffer = GL_DISPATCH_INDIRECT_BUFFER,
		DrawIndirectBuffer = GL_DRAW_INDIRECT_BUFFER,
		ElementBuffer = GL_ELEMENT_ARRAY_BUFFER,
		PixelPackBuffer = GL_PIXEL_PACK_BUFFER,
		PixelUnpackBuffer = GL_PIXEL_UNPACK_BUFFER,
		QueryResultBuffer = GL_QUERY_BUFFER,
		ShaderStorageBuffer = GL_SHADER_STORAGE_BUFFER,			//binding
		TextureBuffer = GL_TEXTURE_BUFFER,
		TransformFeedbackBuffer = GL_TRANSFORM_FEEDBACK_BUFFER,	//binding
		UniformBuffer = GL_UNIFORM_BUFFER,						//binding
	};
	template<ShaderType _shaderType>struct Shader
	{
		GLuint shader;
		String<char>source;

		Shader();
		Shader(String<char>const&);
		void init();
		void create();
		void getSource();
		void compile()const;
		void check()const;
		void omit();
	};
	template<BufferType _vboType>struct Buffer
	{
		struct Data
		{
			enum Usage
			{
				StreamDraw = GL_STREAM_DRAW,
				StreamRead = GL_STREAM_READ,
				StreamCopy = GL_STREAM_COPY,
				StaticDraw = GL_STATIC_DRAW,
				StaticRead = GL_STATIC_READ,
				StaticCopy = GL_STATIC_COPY,
				DynamicDraw = GL_DYNAMIC_DRAW,
				DynamicRead = GL_DYNAMIC_READ,
				DynamicCopy = GL_DYNAMIC_COPY
			};
			Usage usage;
			virtual void* pointer() = 0;
			virtual unsigned int size() = 0;
			Data();
			Data(Usage);
		};
		//Then other data types inherit from this.
		Data* data;
		GLuint buffer;

		Buffer();
		Buffer(Data*);
		void create();
		void bind();
		static void unbind();
		void dataStore();
	};
	template<>struct Buffer<AtomicCounterBuffer>
	{
		struct Data
		{
			GLuint usage;
			virtual void* pointer() = 0;
			virtual unsigned int size() = 0;
		};
		//Then other data types inherit from this.
		Data* data;
		GLuint buffer;
		unsigned int binding;

		Buffer()
			:
			data(nullptr),
			buffer(0)
		{

		}
		Buffer(Data* _data, unsigned int _index)
			:
			data(_data),
			binding(_index)
		{

		}
		void init()
		{
			create();
			bind();
			bindBase();
		}
		void create()
		{
			glCreateBuffers(1, &buffer);
		}
		void bind()
		{
			glBindBuffer(AtomicCounterBuffer, buffer);
		}
		void bindBase()
		{
			glBindBufferBase(AtomicCounterBuffer, binding, buffer);
		}
		static void unbind()
		{
			glBindBuffer(AtomicCounterBuffer, 0);
		}
		void dataStore()
		{
			glBufferData(AtomicCounterBuffer, data->size(), data->pointer(), data->usage);
		}
	};
	template<>struct Buffer<ShaderStorageBuffer>
	{
		struct Data
		{
			enum Usage
			{
				StreamDraw = GL_STREAM_DRAW,
				StreamRead = GL_STREAM_READ,
				StreamCopy = GL_STREAM_COPY,
				StaticDraw = GL_STATIC_DRAW,
				StaticRead = GL_STATIC_READ,
				StaticCopy = GL_STATIC_COPY,
				DynamicDraw = GL_DYNAMIC_DRAW,
				DynamicRead = GL_DYNAMIC_READ,
				DynamicCopy = GL_DYNAMIC_COPY
			};
			Usage usage;
			virtual void* pointer() = 0;
			virtual unsigned int size() = 0;
			Data()
				:
				usage(StaticDraw)
			{

			}
			Data(Usage _usage)
				:
				usage(_usage)
			{

			}
		};
		//Then other data types inherit from this.
		Data* data;
		GLuint buffer;
		unsigned int binding;

		Buffer()
			:
			data(nullptr),
			buffer(0)
		{

		}
		Buffer(Data* _data, unsigned int _index)
			:
			data(_data),
			binding(_index)
		{
			create();
			bind();
			bindBase();
		}
		void create()
		{
			glCreateBuffers(1, &buffer);
		}
		void bind()
		{
			glBindBuffer(ShaderStorageBuffer, buffer);
		}
		void bindBase()
		{
			glBindBufferBase(ShaderStorageBuffer, binding, buffer);
		}
		static void unbind()
		{
			glBindBuffer(ShaderStorageBuffer, 0);
		}
		void dataStore()
		{
			glBufferData(ShaderStorageBuffer, data->size(), data->pointer(), data->usage);
		}
	};
	template<>struct Buffer<TransformFeedbackBuffer>
	{
		struct Data
		{
			GLuint usage;
			virtual void* pointer() = 0;
			virtual unsigned int size() = 0;
		};
		//Then other data types inherit from this.
		Data* data;
		GLuint buffer;
		unsigned int binding;

		Buffer()
			:
			data(nullptr),
			buffer(0)
		{

		}
		Buffer(Data* _data, unsigned int _index)
			:
			data(_data),
			binding(_index)
		{

		}
		void init()
		{
			create();
			bind();
			bindBase();
		}
		void create()
		{
			glCreateBuffers(1, &buffer);
		}
		void bind()
		{
			glBindBuffer(TransformFeedbackBuffer, buffer);
		}
		void bindBase()
		{
			glBindBufferBase(TransformFeedbackBuffer, binding, buffer);
		}
		static void unbind()
		{
			glBindBuffer(TransformFeedbackBuffer, 0);
		}
		void dataStore()
		{
			glBufferData(TransformFeedbackBuffer, data->size(), data->pointer(), data->usage);
		}
	};
	template<>struct Buffer<UniformBuffer>
	{
		struct Data
		{
			enum Usage
			{
				StreamDraw = GL_STREAM_DRAW,
				StreamRead = GL_STREAM_READ,
				StreamCopy = GL_STREAM_COPY,
				StaticDraw = GL_STATIC_DRAW,
				StaticRead = GL_STATIC_READ,
				StaticCopy = GL_STATIC_COPY,
				DynamicDraw = GL_DYNAMIC_DRAW,
				DynamicRead = GL_DYNAMIC_READ,
				DynamicCopy = GL_DYNAMIC_COPY
			};
			Usage usage;
			virtual void* pointer() = 0;
			virtual unsigned int size() = 0;
			Data()
				:
				usage(StaticDraw)
			{

			}
			Data(Usage _usage)
				:
				usage(_usage)
			{

			}
		};
		//Then other data types inherit from this.
		Data* data;
		GLuint buffer;
		unsigned int binding;

		Buffer()
			:
			data(nullptr)
		{
			create();
		}
		Buffer(Data* _data, unsigned int _index)
			:
			data(_data),
			binding(_index)
		{
			create();
			bind();
			bindBase();
		}
		void create()
		{
			glCreateBuffers(1, &buffer);
		}
		void bind()
		{
			glBindBuffer(UniformBuffer, buffer);
		}
		void bindBase()
		{
			glBindBufferBase(UniformBuffer, binding, buffer);
		}
		static void unbind()
		{
			glBindBuffer(UniformBuffer, 0);
		}
		void dataStore()
		{
			bind();
			glBufferData(UniformBuffer, data->size(), data->pointer(), data->usage);
		}
		void refreshData()
		{
			bind();
			glBufferSubData(UniformBuffer, 0, data->size(), data->pointer());
		}
	};
	struct VertexAttrib
	{
		enum Size
		{
			one = 1,
			two = 2,
			three = 3,
			four = 4,
			BGRA = GL_BGRA,
		};
		enum Type
		{
			Byte = GL_BYTE,
			UByte = GL_UNSIGNED_BYTE,
			Short = GL_SHORT,
			UShort = GL_UNSIGNED_SHORT,
			Int = GL_INT,
			UInt = GL_UNSIGNED_INT,
			HalfFloat = GL_HALF_FLOAT,
			Float = GL_FLOAT,
			Doule = GL_DOUBLE,
			Fixed = GL_FIXED,
			Int_2_10_10_10 = GL_INT_2_10_10_10_REV,
			UINT_2_10_10_10 = GL_UNSIGNED_INT_2_10_10_10_REV,
			UINT_10_11_11 = GL_UNSIGNED_INT_10F_11F_11F_REV,
		};

		Buffer<ArrayBuffer>* buffer;
		unsigned int binding;
		Size size;
		Type type;
		bool normalized;
		int stride;
		int offset;
		int divisor;

		VertexAttrib();
		VertexAttrib(Buffer<ArrayBuffer>*, unsigned int, Size, Type, bool, int, int, int);
		void init();
		void bind();
		void enable()const;
		void disable()const;
	};
	struct VertexArrayBuffer
	{
		GLuint vao;
		Vector<VertexAttrib*>attribs;
		VertexArrayBuffer();
		VertexArrayBuffer(Vector<VertexAttrib*>const&);
		void init();
		void bind();
		static void unbind();
	};
	struct ShaderManager
	{
		Vector<Shader<VertexShader>> vertex;
		Vector<Shader<TessControlShader>> tessControl;
		Vector<Shader<TessEvaluationShader>> tessEvaluation;
		Vector<Shader<GeometryShader>> geometry;
		Vector<Shader<FragmentShader>> fragment;
		Vector<Shader<ComputeShader>> compute;
		ShaderManager() = default;
		ShaderManager(Array<Vector<String<char>>, 6>const&);
		void init();
		void omit();
	};
	struct SourceManager
	{
		struct Source
		{
			using Attach = Vector<Pair<unsigned int, unsigned int>>;
			String<char>name;
			Array<Vector<String<char>>, 6>source;
			Attach attach;
			Source() = delete;
			Source(String<char>const&);
			bool addSource(char const*, String<char>const&);
		};

		Vector<Source> sources;
		File folder;

		SourceManager();
		SourceManager(String<char> const&);
		void readSource();
		void deleteSource();
		Source& getProgram(String<char>const&);
	};
	struct Program
	{
		using Attach = Vector<Pair<unsigned int, unsigned int>>;

		SourceManager* sourceManage;
		String<char> name;
		unsigned int num;
		ShaderManager shaders;
		GLuint program;
		VertexArrayBuffer vao;

		Program(SourceManager*, String<char>const&);
		Program(SourceManager*, String<char>const&, Vector <VertexAttrib*>const&);
		void init();
		void create();
		void attach();
		void link();
		void check();
		void use();
		virtual void setBufferData() = 0;
		virtual void run() = 0;
		virtual void resize(int, int) = 0;
		//In order to run:
		//	init the buffers
		//	init();
		//	use();
		//	run();
	};
	struct Transform
	{
		struct Data
		{
			struct Perspective
			{
				double fovy;
				double zNear;
				double zFar;
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
			double aspect;
			double fovy;
			double zNear;
			double zFar;
			bool updated;
			Perspective();
			Perspective(Data::Perspective const&);
			Perspective(Perspective const&) = default;
			void init(FrameScale const&);
			void refresh(int, int);
		};
		struct Scroll
		{
			double increaseDelta;
			double decreaseRatio;
			double threshold;
			double total;
			Scroll();
			Scroll(Data::Scroll const&);
			void refresh(double);
			double operate();
		};
		struct Key
		{
			bool left;
			bool right;
			bool up;
			bool down;
			double ratio;
			Key();
			Key(Data::Key const&);
			void refresh(int, bool);
			Math::vec2<double>operate();
		};
		struct Mouse
		{
			struct Pointer
			{
				double x;
				double y;
				bool valid;
				Pointer();
			};
			Pointer now;
			Pointer pre;
			bool left;
			bool middle;
			bool right;
			Mouse();
			void refreshPos(double, double);
			void refreshButton(int, bool);
			Math::vec2<double> operate();
		};
		struct BufferData :Buffer<UniformBuffer>::Data
		{
			Math::mat4<float>ans;
			BufferData();
			virtual void* pointer()override
			{
				return (void*)(ans.array);
			}
			virtual unsigned int size()override
			{
				return sizeof(ans);
			}
		};


		Perspective persp;
		Scroll scroll;
		Key key;
		Mouse mouse;
		BufferData bufferData;
		Math::vec3<double>dr;
		Math::mat4<double>proj;
		Math::mat4<double>trans;
		double depth;
		bool updated;

		Transform();
		Transform(Data const&);
		void init(FrameScale const&);
		void resize(int, int);
		void calcProj();
		void calcAns();
		void operate();
	};




	//OpenGLInit
	bool OpenGLInit::initialized(false);
	unsigned int OpenGLInit::versionMajor(4);
	unsigned int OpenGLInit::versionMinor(5);

	inline OpenGLInit::OpenGLInit()
	{
		if (!initialized)
		{
			glfwInit();
			setOpenGLVersion(4, 5);
			glewExperimental = GL_TRUE;
			initialized = true;
		}
	}
	inline OpenGLInit::OpenGLInit(unsigned int _major, unsigned int _minor)
	{
		if (!initialized)
		{
			glfwInit();
			glewExperimental = GL_TRUE;
			setOpenGLVersion(_major, _minor);
			initialized = true;
		}
	}
	inline void OpenGLInit::setOpenGLVersion(unsigned int _major, unsigned int _minor)
	{
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, versionMajor = _major);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, versionMinor = _minor);
	}

	template<ShaderType _shaderType>inline Shader<_shaderType>::Shader()
		:
		shader(0),
		source()
	{
	}
	template<ShaderType _shaderType>inline Shader<_shaderType>::Shader(String<char> const& _source)
		:
		source(_source)
	{
		create();
		getSource();
	}
	template<ShaderType _shaderType>inline void Shader<_shaderType>::init()
	{
		if (!shader)
		{
			create();
			getSource();
		}
		compile();
		check();
	}
	template<ShaderType _shaderType>inline void Shader<_shaderType>::create()
	{
		shader = glCreateShader(_shaderType);
	}
	template<ShaderType _shaderType>inline void Shader<_shaderType>::getSource()
	{
		glShaderSource(shader, 1, &source.data, NULL);
	}
	template<ShaderType _shaderType>inline void Shader<_shaderType>::compile() const
	{
		glCompileShader(shader);
	}
	template<ShaderType _shaderType>inline void Shader<_shaderType>::check() const
	{
		char log[1024];
		GLint success(1);
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 1024, NULL, log);
			::printf("%s\n", log);
			exit(-1);
		}
	}
	template<ShaderType _shaderType>inline void Shader<_shaderType>::omit()
	{
		glDeleteShader(shader);
	}

	template<BufferType _vboType>inline Buffer<_vboType>::Buffer()
		:
		data(nullptr)
	{
		create();
	}
	template<BufferType _vboType>inline Buffer<_vboType>::Buffer(Data * _data)
		:
		data(_data)
	{
		create();
	}
	template<BufferType _vboType>inline void Buffer<_vboType>::create()
	{
		glCreateBuffers(1, &buffer);
	}
	template<BufferType _vboType>inline void Buffer<_vboType>::bind()
	{
		glBindBuffer(_vboType, buffer);
	}
	template<BufferType _vboType>inline void Buffer<_vboType>::unbind()
	{
		glBindBuffer(_vboType, 0);
	}
	template<BufferType _vboType>inline void Buffer<_vboType>::dataStore()
	{
		bind();
		glBufferData(_vboType, data->size(), data->pointer(), data->usage);
	}

	inline VertexAttrib::VertexAttrib()
		:
		binding(0),
		size(four),
		type(Float),
		normalized(false),
		stride(0),
		offset(0),
		buffer(nullptr)
	{
	}
	inline VertexAttrib::VertexAttrib(Buffer<ArrayBuffer> * _buffer,
		unsigned int _index, Size _size, Type _type,
		bool _normalized, int _stride, int _offset, int _divisor)
		:
		buffer(_buffer),
		binding(_index),
		size(_size),
		type(_type),
		normalized(_normalized),
		stride(_stride),
		offset(_offset),
		divisor(_divisor)
	{
	}
	inline void VertexAttrib::init()
	{
		buffer->bind();
		enable();
		glVertexAttribPointer(binding, size, type, normalized, stride, (void const*)offset);
		if (divisor)
			glVertexAttribDivisor(binding, divisor);
	}
	inline void VertexAttrib::bind()
	{
		buffer->bind();
	}
	inline void VertexAttrib::enable()const
	{
		glEnableVertexAttribArray(binding);
	}
	inline void VertexAttrib::disable()const
	{
		glDisableVertexAttribArray(binding);
	}


	inline ShaderManager::ShaderManager(Array<Vector<String<char>>, 6> const& _sources)
		:
		vertex(_sources.data[0]),
		tessControl(_sources.data[1]),
		tessEvaluation(_sources.data[2]),
		geometry(_sources.data[3]),
		fragment(_sources.data[4]),
		compute(_sources.data[5])
	{
	}
	inline void ShaderManager::init()
	{
		vertex.traverse([](Shader<VertexShader> & _shader)
			{
				_shader.init();
				return true;
			});
		tessControl.traverse([](Shader<TessControlShader> & _shader)
			{
				_shader.init();
				return true;
			});
		tessEvaluation.traverse([](Shader<TessEvaluationShader> & _shader)
			{
				_shader.init();
				return true;
			});
		geometry.traverse([](Shader<GeometryShader> & _shader)
			{
				_shader.init();
				return true;
			});
		fragment.traverse([](Shader<FragmentShader> & _shader)
			{
				_shader.init();
				return true;
			});
		compute.traverse([](Shader<ComputeShader> & _shader)
			{
				_shader.init();
				return true;
			});
	}
	inline void ShaderManager::omit()
	{
		vertex.traverse([](Shader<VertexShader> & _shader)
			{
				_shader.omit();
				return true;
			});
		tessControl.traverse([](Shader<TessControlShader> & _shader)
			{
				_shader.omit();
				return true;
			});
		tessEvaluation.traverse([](Shader<TessEvaluationShader> & _shader)
			{
				_shader.omit();
				return true;
			});
		geometry.traverse([](Shader<GeometryShader> & _shader)
			{
				_shader.omit();
				return true;
			});
		fragment.traverse([](Shader<FragmentShader> & _shader)
			{
				_shader.omit();
				return true;
			});
		compute.traverse([](Shader<ComputeShader> & _shader)
			{
				_shader.omit();
				return true;
			});
	}



	inline VertexArrayBuffer::VertexArrayBuffer()
		:
		vao(0)
	{
	}
	inline VertexArrayBuffer::VertexArrayBuffer(Vector<VertexAttrib*> const& _attribs)
		:
		attribs(_attribs)
	{
	}
	inline void VertexArrayBuffer::init()
	{
		glCreateVertexArrays(1, &vao);
		attribs.traverse([](VertexAttrib * &_vertexAttrib)
			{
				_vertexAttrib->bind();
				return true;
			});
		bind();
		attribs.traverse([](VertexAttrib * &_vertexAttrib)
			{
				_vertexAttrib->init();
				return true;
			});
	}
	inline void VertexArrayBuffer::bind()
	{
		glBindVertexArray(vao);
	}
	inline void VertexArrayBuffer::unbind()
	{
		glBindVertexArray(0);
	}



	inline Program::Program(SourceManager * _sourceManage, String<char>const& _name)
		:
		sourceManage(_sourceManage),
		name(_name),
		num((&_sourceManage->getProgram(name) - _sourceManage->sources.data)),
		shaders(_sourceManage->sources[num].source),
		program(0),
		vao()
	{
	}
	inline Program::Program(SourceManager * _sourceManage, String<char>const& _name, Vector<VertexAttrib*>const& _attribs)
		:
		sourceManage(_sourceManage),
		name(_name),
		num((&_sourceManage->getProgram(name) - _sourceManage->sources.data)),
		shaders(_sourceManage->sources[num].source),
		program(0),
		vao(_attribs)
	{
	}
	inline void Program::init()
	{
		shaders.init();
		create();
		attach();
		link();
		check();
		vao.init();
	}
	inline void Program::create()
	{
		program = glCreateProgram();
	}
	inline void Program::attach()
	{
		Attach const& attach(sourceManage->sources[num].attach);
		for (int c0(0); c0 < attach.length; ++c0)
			switch (attach.data[c0].data0)
			{
				case 0:glAttachShader(program, shaders.vertex[attach.data[c0].data1].shader); break;
				case 1:glAttachShader(program, shaders.tessControl[attach.data[c0].data1].shader); break;
				case 2:glAttachShader(program, shaders.tessEvaluation[attach.data[c0].data1].shader); break;
				case 3:glAttachShader(program, shaders.geometry[attach.data[c0].data1].shader); break;
				case 4:glAttachShader(program, shaders.fragment[attach.data[c0].data1].shader); break;
				case 5:glAttachShader(program, shaders.compute[attach.data[c0].data1].shader); break;
			}
	}
	inline void Program::link()
	{
		glLinkProgram(program);
	}
	inline void Program::check()
	{
		char log[1024];
		GLint success(1);
		glGetProgramiv(program, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(program, 2014, NULL, log);
			exit(-1);
		}
	}
	inline void Program::use()
	{
		glUseProgram(program);
		vao.bind();
		setBufferData();
	}




	inline SourceManager::SourceManager()
		:
		folder("./"),
		sources()
	{
		readSource();
	}
	inline SourceManager::SourceManager(String<char> const& _path)
		:
		folder(_path),
		sources()
	{
		readSource();
	}
	inline void SourceManager::readSource()
	{
		File& shaders(folder.find("shaders"));
		String<char>ShaderLists(shaders.findInThis("ShaderLists.txt").readText());
		Vector<int>programs(ShaderLists.find("Program:"));
		char const* table[6] =
		{
			"Vertex",
			"TessControl",
			"TessEvaluation",
			"Geometry",
			"Fragment",
			"Compute"
		};
		for (int c0(0); c0 < programs.length; ++c0)
		{
			char t0[100];
			char t1[100];
			char t2[5];
			int n(programs[c0]);
			int delta(0);
			sscanf(ShaderLists.data + n, "Program:%n", &delta);
			n += delta;
			n += sweepStr(ShaderLists.data + n, "%*[^a-zA-Z0-9_]%n");
			sscanf(ShaderLists.data + n, "%[a-zA-Z0-9_]%n", t0, &delta);
			n += delta;
			n += sweepStr(ShaderLists.data + n, "%*[^{]{%n");
			String<char>program(t0);
			sources.pushBack(program);
			int s(0);
			do
			{
				s = sscanf(ShaderLists.data + n, "%s%s%n%*[\t\r\n ]%[}]", t0, t1, &delta, t2);
				n += delta;
				if (s < 2)break;
				if (!sources.end().addSource(t0, shaders.findInThis(program + t0 + t1 + ".cpp").readText()))
					::printf("Cannot read Program: %s\n", program.data);
			}
			while (s == 2);
		}
	}
	inline void SourceManager::deleteSource()
	{
		(&folder)->~File();
		(&sources)->~Vector();
	}
	inline SourceManager::Source& SourceManager::getProgram(String<char>const& _name)
	{
		for (int c0(0); c0 < sources.length; ++c0)
			if (sources.data[c0].name == _name)
				return sources.data[c0];
		return *(Source*)NULL;
	}

	inline SourceManager::Source::Source(String<char> const& _name)
		:
		name(_name)
	{
	}
	inline bool SourceManager::Source::addSource(char const* _type, String<char>const& _source)
	{
		char const* table[6] =
		{
			"Vertex",
			"TessControl",
			"TessEvaluation",
			"Geometry",
			"Fragment",
			"Compute"
		};
		int c0(0);
		for (; c0 < 6; ++c0)
			if (!strcmp(_type, table[c0]))break;
		if (c0 < 6)
		{
			source.data[c0].pushBack(_source);
			attach.pushBack(Pair<unsigned int, unsigned int>(c0, source.data[c0].length - 1));
			return true;
		}
		return false;
	}

	template<BufferType _vboType>inline Buffer<_vboType>::Data::Data()
		:
		usage(StaticDraw)
	{
	}
	template<BufferType _vboType>inline Buffer<_vboType>::Data::Data(Usage _usage)
		:
		usage(_usage)
	{
	}

	inline Transform::Perspective::Perspective()
		:
		aspect(16.0 / 9.0),
		fovy(Math::Pi * 100.0 / 180.0),
		zNear(0.1),
		zFar(100),
		updated(false)
	{
	}
	inline Transform::Perspective::Perspective(Data::Perspective const& _persp)
		:
		aspect(16.0 / 9.0),
		fovy(_persp.fovy),
		zNear(_persp.zNear),
		zFar(_persp.zFar),
		updated(false)
	{
	}
	inline void Transform::Perspective::init(FrameScale const& _size)
	{
		aspect = double(_size.w) / _size.h;
		updated = false;
	}
	inline void Transform::Perspective::refresh(int _w, int _h)
	{
		aspect = double(_w) / double(_h);
		updated = true;
	}

	inline Transform::Scroll::Scroll()
		:
		increaseDelta(0.05),
		decreaseRatio(0.95),
		threshold(0.01),
		total(threshold)
	{
	}
	inline Transform::Scroll::Scroll(Data::Scroll const& _scroll)
		:
		increaseDelta(_scroll.increaseDelta),
		decreaseRatio(_scroll.decreaseRatio),
		threshold(_scroll.threshold),
		total(threshold)
	{
	}
	inline void Transform::Scroll::refresh(double _d)
	{
		total += _d * increaseDelta;
	}
	inline double Transform::Scroll::operate()
	{
		if (abs(total) > threshold)
		{
			total *= decreaseRatio;
			return total;
		}
		else return 0.0;
	}

	inline Transform::Key::Key()
		:
		left(false),
		right(false),
		up(false),
		down(false),
		ratio(0.05)
	{
	}
	inline Transform::Key::Key(Data::Key const& _key)
		:
		left(false),
		right(false),
		up(false),
		down(false),
		ratio(_key.ratio)
	{
	}
	inline void Transform::Key::refresh(int _key, bool _operation)
	{
		switch (_key)
		{
			case 0:left = _operation; break;
			case 1:right = _operation; break;
			case 2:up = _operation; break;
			case 3:down = _operation; break;
		}
	}
	inline Math::vec2<double> Transform::Key::operate()
	{
		Math::vec2<double>t
		{
			ratio * ((int)right - (int)left),
			ratio * ((int)up - (int)down)
		};
		return t;
	}

	inline Transform::Mouse::Pointer::Pointer()
		:
		valid(false)
	{
	}
	inline Transform::Mouse::Mouse()
		:
		now(),
		pre(),
		left(false),
		middle(false),
		right(false)
	{
	}
	inline void Transform::Mouse::refreshPos(double _x, double _y)
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
	inline void Transform::Mouse::refreshButton(int _button, bool _operation)
	{
		switch (_button)
		{
			case 0:	left = _operation; break;
			case 1:	middle = _operation; break;
			case 2:	right = _operation; break;
		}

	}
	inline Math::vec2<double> Transform::Mouse::operate()
	{
		if (now.valid && pre.valid)
		{
			pre.valid = false;
			return { pre.y - now.y ,pre.x - now.x };
		}
		else return { 0.0,0.0 };
	}

	inline Transform::BufferData::BufferData()
		:Data(DynamicDraw)
	{
	}

	inline Transform::Transform()
		:
		persp(),
		scroll(),
		mouse(),
		updated(false),
		bufferData(),
		dr(0.0),
		trans(Math::mat4<double>::id()),
		depth(500.0)
	{
	}
	inline Transform::Transform(Data const& _data)
		:
		persp(_data.persp),
		scroll(_data.scroll),
		key(_data.key),
		mouse(),
		updated(false),
		bufferData(),
		dr(0.0),
		trans(Math::mat4<double>::id()),
		depth(_data.depth)
	{
	}
	inline void Transform::init(FrameScale const& _size)
	{
		persp.init(_size);
		calcProj();
		calcAns();
		updated = true;
	}
	inline void Transform::resize(int _w, int _h)
	{
		persp.refresh(_w, _h);
	}
	inline void Transform::calcProj()
	{
		double cotangent = 1 / tan(Math::Pi * persp.fovy / 360.0);
		proj = 0;
		proj.array[0][0] = cotangent / persp.aspect;
		proj.array[1][1] = cotangent;
		proj.array[2][2] = (persp.zFar + persp.zNear) / (persp.zNear - persp.zFar);
		proj.array[2][3] = 2.0 * persp.zNear * persp.zFar / (persp.zNear - persp.zFar);
		proj.array[3][2] = -1.0;
	}
	inline void Transform::calcAns()
	{
		bufferData.ans = (proj, trans);
	}
	inline void Transform::operate()
	{
		Math::vec3<double>dxyz(key.operate());
		dxyz.data[2] = -scroll.operate();
		Math::vec2<double>axis(mouse.operate());
		bool operated(false);
		if (dxyz != 0.0)
		{
			dr -= dxyz;
			trans.setCol(dr, 3);
			operated = true;
		}
		if (axis != 0.0)
		{
			trans = (Math::vec3<double>(axis).rotMat(axis.length() / depth), trans);
			dr = trans.column(3);
			operated = true;
		}
		if (persp.updated)
		{
			persp.updated = false;
			calcProj();
			operated = true;
		}
		if (operated)
		{
			calcAns();
			updated = true;
		}
	}
}
