#pragma once
#include <GL/_OpenGL.h>
#include <_String.h>
#include <_List.h>

namespace Window
{
	struct Window
	{
		struct CallbackFun
		{
			struct Frame
			{
				GLFWframebuffersizefun size;
				GLFWwindowposfun pos;
				GLFWwindowfocusfun focus;
			};
			struct Input
			{
				GLFWmousebuttonfun button;
				GLFWcursorposfun pos;
				GLFWscrollfun scroll;
				GLFWkeyfun key;
			};
			Frame frame;
			Input input;
		};
		struct Data
		{
			struct Size
			{
				OpenGL::FrameScale size;
				bool resizable;
				bool fullScreen;
			};
			char const* title;
			Size size;
		};

		struct Title
		{
			String<char> title;
			Title() = default;
			Title(char const*);
			bool operator==(char const*)const;
			bool operator==(String<char> const&)const;
			void init(GLFWwindow*);
		};
		struct Monitor
		{
			GLFWmonitor* monitor;
			GLFWmonitor** monitors;
			GLFWvidmode mode;
			String<char>name;
			int num;
			int numAll;

			Monitor();
			void init();
			bool search(int, int);
			String<char>& getName();
		};
		struct Size
		{
			struct FullScreen
			{
				bool fullScreen;
				FullScreen();
				FullScreen(bool);
			};

			bool resizable;
			OpenGL::FrameScale size;
			FullScreen fullScreen;
			Size(OpenGL::FrameScale);
			Size(OpenGL::FrameScale, bool, bool);
			void set(GLFWwindow*, int, int);
		};
		struct Callback
		{
			struct Frame
			{
				Window::CallbackFun::Frame frame;
				Frame() = delete;
				Frame(Window::CallbackFun::Frame const&);
				void init(GLFWwindow*);
			};
			struct Input
			{
				Window::CallbackFun::Input input;
				Input() = delete;
				Input(Window::CallbackFun::Input const&);
				void init(GLFWwindow*);
			};
			Frame frame;
			Input input;
			Callback() = delete;
			Callback(
				Window::CallbackFun::Frame const&,
				Window::CallbackFun::Input const&);
			void init(GLFWwindow*);
		};

		static bool glewInitialized;

		GLFWwindow* window;
		Title title;
		Monitor monitor;
		Size size;
		Callback callback;
		OpenGL::OpenGL* openGL;

		Window() = delete;
		Window(Window const&) = default;
		Window(Data const&, CallbackFun const&);
		bool operator==(GLFWwindow* const)const;
		void init(OpenGL::OpenGL*);
		void setTitle(char const*);
	};
	bool Window::glewInitialized(false);

	void frameSizeCallback(GLFWwindow*, int _w, int _h);
	void framePosCallback(GLFWwindow*, int _w, int _h);
	void frameFocusCallback(GLFWwindow*, int _focused);

	void mouseButtonCallback(GLFWwindow*, int _button, int _action, int _mods);
	void mousePosCallback(GLFWwindow*, double _x, double _y);
	void mouseScrollCallback(GLFWwindow*, double _x, double _y);
	void keyCallback(GLFWwindow*, int _key, int _scancode, int _action, int _mods);



	struct WindowManager
	{
		static constexpr Window::CallbackFun callbackFun
		{
			{
				frameSizeCallback,
				framePosCallback,
				frameFocusCallback
			},
			{
				mouseButtonCallback,
				mousePosCallback,
				mouseScrollCallback,
				keyCallback
			}
		};

		List<Window> windows;

		WindowManager();
		WindowManager(Window::Data const&);
		void init(unsigned int, OpenGL::OpenGL*);
		void createWindow(Window::Data const&);
		Window& find(GLFWwindow* const);
		void render();
		void swapBuffers();
		void pullEvents();
		bool close();
	};

	WindowManager* __windowManager = nullptr;


	void frameSizeCallback(GLFWwindow* _window, int _w, int _h)
	{
		__windowManager->find(_window).openGL->frameSize(_w, _h);
	}
	void framePosCallback(GLFWwindow* _window, int _w, int _h)
	{
		__windowManager->find(_window).openGL->framePos(_w, _h);
	}
	void frameFocusCallback(GLFWwindow* _window, int _focused)
	{
		__windowManager->find(_window).openGL->frameFocus(_focused);
	}
	void mouseButtonCallback(GLFWwindow* _window, int _button, int _action, int _mods)
	{
		__windowManager->find(_window).openGL->mouseButton(_button, _action, _mods);
	}
	void mousePosCallback(GLFWwindow* _window, double _x, double _y)
	{
		__windowManager->find(_window).openGL->mousePos(_x, _y);
	}
	void mouseScrollCallback(GLFWwindow* _window, double _x, double _y)
	{
		__windowManager->find(_window).openGL->mouseScroll(_x, _y);
	}
	void keyCallback(GLFWwindow* _window, int _key, int _scancode, int _action, int _mods)
	{
		__windowManager->find(_window).openGL->key(_window, _key, _scancode, _action, _mods);
	}

	//Window
	inline Window::Title::Title(char const* _title)
		:
		title(_title)
	{
	}
	inline bool Window::Title::operator==(char const* a) const
	{
		return title == a;
	}
	inline bool Window::Title::operator==(String<char> const& a) const
	{
		return title == a;
	}
	inline void Window::Title::init(GLFWwindow* _window)
	{
		glfwSetWindowTitle(_window, title.data);
	}

	inline Window::Monitor::Monitor()
		:
		monitor(nullptr),
		monitors(nullptr),
		mode(),
		name(),
		num(-1),
		numAll(0)
	{
	}
	inline void Window::Monitor::init()
	{
		monitors = glfwGetMonitors(&numAll);
	}
	inline bool Window::Monitor::search(int _w, int _h)
	{
		if (!monitors)init();
		for (int c0 = 0; c0 < numAll; ++c0)
		{
			GLFWvidmode const* _mode(glfwGetVideoMode(monitors[c0]));
			if (_mode->width == _w && _mode->height == _h)
			{
				mode = *_mode;
				monitor = monitors[c0];
				num = c0;
				return true;
			}
		}
		mode = *glfwGetVideoMode(monitors[0]);
		monitor = monitors[0];
		num = 0;
		return false;
	}
	inline String<char>& Window::Monitor::getName()
	{
		if (monitor)
			return name = glfwGetMonitorName(monitor);
		else
			return name = "No monitor!!";
	}

	inline Window::Size::FullScreen::FullScreen()
		:
		fullScreen(false)
	{

	}
	inline Window::Size::FullScreen::FullScreen(bool _fullScreen)
		:
		fullScreen(_fullScreen)
	{

	}

	inline Window::Size::Size(OpenGL::FrameScale _size)
		:
		size(size),
		resizable(true),
		fullScreen(false)
	{
		glfwWindowHint(GLFW_RESIZABLE, true);
	}
	inline Window::Size::Size(OpenGL::FrameScale _size, bool _resizable, bool _fullScreen)
		:
		size(_size),
		resizable(_resizable),
		fullScreen(_fullScreen)
	{
		glfwWindowHint(GLFW_RESIZABLE, _resizable);
	}
	inline void Window::Size::set(GLFWwindow* _window, int _w, int _h)
	{
		if (!fullScreen.fullScreen && resizable)
			glfwSetWindowSize(_window, size.w = _w, size.h = _h);
	}



	inline Window::Callback::Frame::Frame(Window::CallbackFun::Frame const& _input)
		:
		frame(_input)
	{
	}
	inline void Window::Callback::Frame::init(GLFWwindow * _window)
	{
		glfwSetFramebufferSizeCallback(_window, frame.size);
		glfwSetWindowPosCallback(_window, frame.pos);
		glfwSetWindowFocusCallback(_window, frame.focus);
	}
	inline Window::Callback::Input::Input(typename Window::CallbackFun::Input const& _input)
		:
		input(_input)
	{
	}
	inline void Window::Callback::Input::init(GLFWwindow * _window)
	{
		glfwSetMouseButtonCallback(_window, input.button);
		glfwSetCursorPosCallback(_window, input.pos);
		glfwSetScrollCallback(_window, input.scroll);
		glfwSetKeyCallback(_window, input.key);
	}

	inline Window::Callback::Callback(
		Window::CallbackFun::Frame const& _frameIn,
		Window::CallbackFun::Input const& _iinputIn)
		:
		frame(_frameIn),
		input(_iinputIn)
	{
	}
	inline void Window::Callback::init(GLFWwindow * _window)
	{
		frame.init(_window);
		input.init(_window);
	}

	inline Window::Window(Data const& _data, ::Window::Window::CallbackFun const& _callback)
		:
		window(nullptr),
		title(_data.title),
		monitor(),
		size(_data.size.size, _data.size.resizable, _data.size.fullScreen),
		callback(_callback.frame, _callback.input),
		openGL(nullptr)
	{
		if (size.fullScreen.fullScreen)
		{
			monitor.search(size.size.w, size.size.h);
			window = glfwCreateWindow(
				size.size.w = monitor.mode.width,
				size.size.h = monitor.mode.height,
				title.title, monitor.monitor,
				NULL);
		}
		else
			window = glfwCreateWindow(size.size.w, size.size.h, title.title, NULL, NULL);
		glfwMakeContextCurrent(window);
		if (!glewInitialized)
		{
			glewInitialized = true;
			glewInit();
		}
		//callback.init(window);
		//openGL->init(size.size);
	}
	inline bool Window::operator==(GLFWwindow * const _window) const
	{
		return window == _window;
	}
	inline void Window::init(OpenGL::OpenGL * _openGL)
	{
		openGL = _openGL;
		callback.init(window);
		openGL->init(size.size);
	}
	inline void Window::setTitle(char const* _title)
	{
		glfwSetWindowTitle(window, _title);
	}

	//WindowManager
	inline WindowManager::WindowManager()
	{
		__windowManager = this;
	}
	inline WindowManager::WindowManager(Window::Data const& _data)
		:
		windows((__windowManager = this, Window(_data, callbackFun)))
	{
	}
	inline void WindowManager::init(unsigned int _num, OpenGL::OpenGL * _openGL)
	{
		windows[_num].data.init(_openGL);
	}
	inline void WindowManager::createWindow(Window::Data const& _data)
	{
		windows.pushBack(Window(_data, callbackFun));
	}
	inline Window& WindowManager::find(GLFWwindow * const _window)
	{
		return windows.find(_window).data;
	}

	inline void WindowManager::render()
	{
		windows.traverse
		([](Window const& _window)
			{
				glfwMakeContextCurrent(_window.window);
				_window.openGL->run();
				return true;
			}
		);
	}
	inline void WindowManager::swapBuffers()
	{
		windows.traverse
		([](Window const& _window)
			{
				glfwMakeContextCurrent(_window.window);
				glfwSwapBuffers(_window.window);
				return true;
			}
		);
	}
	inline void WindowManager::pullEvents()
	{
		glfwPollEvents();
	}
	inline bool WindowManager::close()
	{
		windows.check([](Window const& _window)
			{
				if (glfwWindowShouldClose(_window.window))
				{
					glfwDestroyWindow(_window.window);
					return false;
				}
				return true;
			});
		if (!windows.length)return true;
		return false;
	}
}
