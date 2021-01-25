#pragma once
#include <_KRPC/_KRPC.h>
#include <_Time.h>

namespace KRPC
{
	struct KTimer
	{
		timespec beginning;
		timespec ending;
		//unit: ms
		double time;
		void begin()
		{
			timespec_get(&beginning, TIME_UTC);
		}
		void end()
		{
			timespec_get(&ending, TIME_UTC);
		}
		double getTime()
		{
			timespec_get(&ending, TIME_UTC);
			time = double(ending.tv_nsec - beginning.tv_nsec) / 1000000;
			time += 1000ll * (ending.tv_sec - beginning.tv_nsec);
			return time;
		}
		void wait(double ms)
		{
			timespec tp;
			timespec_get(&tp, TIME_UTC);
			long long _dt(ms * 1000000ll), dt;
			do
			{
				end();
				dt = 1000000000LL * (ending.tv_sec - tp.tv_sec) + (ending.tv_nsec - tp.tv_nsec);
			} while (dt < _dt);
		}
		void print()
		{
			printf("%lf\n", getTime());
		}
	};
	static KTimer Time;
}