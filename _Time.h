#pragma once
#include <ctime>
#include <cstdio>
#include <cstring>
#include <cmath>

struct Timer
{
	timespec begining;
	timespec ending;
	Timer() = default;
	void begin();
	void end();
	void print();
	void print(const char* a);
};
struct FPS
{
	timespec t0;
	timespec t1;
	unsigned long long dt;
	double fps;
	char str[128];
	bool valid;
	FPS();
	void refresh();
	void printFPS(unsigned int a);
	void printFPSToString(unsigned int a);
	void printFrameTime(unsigned int a);
	void printFrameTimeToString(unsigned int a);
	void printFPSAndFrameTime(unsigned int a, unsigned int b);
	void printFPSAndFrameTimeToString(unsigned int a, unsigned int b);
};





inline void Timer::begin()
{
	timespec_get(&begining, TIME_UTC);
}
inline void Timer::end()
{
	timespec_get(&ending, TIME_UTC);
}
inline void Timer::print()
{
	constexpr unsigned long long e3{ 1000U };
	constexpr unsigned long long e6{ 1000000U };
	constexpr unsigned long long e9{ 1000000000U };
	constexpr unsigned long long e12{ 100000000000U };
	long long dt = e9 * (ending.tv_sec - begining.tv_sec) + (ending.tv_nsec - begining.tv_nsec);
	if (dt < 0)printf("-");
	unsigned long long dt1 = llabs(dt);
	if (dt1 >= 500)dt1 -= 500;
	else dt1 = 0;
	double length = log10(dt1);
	if (length < 3)printf("%15llu ns", dt1);
	else if (length < 6)printf("%16llu,%03llu ns", dt1 / e3, dt1 % e3);
	else if (length < 9)printf("%12llu,%03llu,%03llu ns", dt1 / e6, (dt1 % e6) / e3, dt1 % e3);
	else if (length < 12)printf("%8llu,%03llu,%03llu,%03llu ns", dt1 / e9, (dt1 % e9) / e6, (dt1 % e6) / e3, dt1 % e3);
	else if (length < 15)printf("%4llu,%03llu,%03llu,%03llu,%03llu ns", dt1 / e12, (dt1 % e12) / e9, (dt1 % e6) / e6, (dt1 % e6) / e6, dt1 % e3);
	printf("\n");
}
inline void Timer::print(const char* a)
{
	printf(a);
	print();
}
//==================FPS==================
inline FPS::FPS()
	:
	valid(false)
{
}
inline void FPS::refresh()
{
	constexpr unsigned long long e9{ 1000000000U };
	if (valid)
	{
		timespec_get(&t1, TIME_UTC);
		dt = e9 * (t1.tv_sec - t0.tv_sec) + t1.tv_nsec - t0.tv_nsec;
		t0 = t1;
		fps = (double)e9 / dt;
	}
	else
	{
		timespec_get(&t0, TIME_UTC);
		valid = true;
	}
}
inline void FPS::printFPS(unsigned int a)
{
	if (valid)printf("\rfps: %.*lf    ", a, fps);
}
inline void FPS::printFPSToString(unsigned int a)
{
	if (valid)sprintf(str, "fps: %.*lf", a, fps);
}
inline void FPS::printFrameTime(unsigned int a)
{
	if (valid)printf("\rframe time: %.*f ms  ", a, dt / 100000.0);
}
inline void FPS::printFrameTimeToString(unsigned int a)
{
	if (valid)sprintf(str,"frame time: %.*f ms", a, dt / 100000.0);
}
inline void FPS::printFPSAndFrameTime(unsigned int a, unsigned int b)
{
	if (valid)printf("\rfps:%.*lf\tframe time: %.*lf ms      ", a, fps, b, dt / 1000000.0);
}
inline void FPS::printFPSAndFrameTimeToString(unsigned int a, unsigned int b)
{
	if (valid)sprintf(str, "fps:%.*lf    frame time: %.*lf ms", a, fps, b, dt / 1000000.0);
}