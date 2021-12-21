#pragma once
#include <_KRPC/_Time.h>

namespace KRPC
{
	struct PID
	{
		double Kp, Ki, Kd;
		double cMin, cMax;
		double seek, P, I, D;
		double t;
		double input;

		void init(double _Kp, double _Ki, double _Kd, double _cMin, double _cMax)
		{
			Kp = _Kp;
			Ki = _Ki;
			Kd = _Kd;
			cMin = _cMin;
			cMax = _cMax;

			seek = 0;
			P = 0;
			I = 0;
			D = 0;
			t = Time.getTime();
			input = 0;
		}
		double newInput(double seek_val, double cur_val)
		{
			double P1(seek_val - cur_val);
			double D1(D);
			double input1(input);
			double t1(Time.getTime());
			double dt(t1 - t);

			if (dt > 0)
			{
				D1 = (P1 - P) / dt;
				double only_pd(Kp * P1 + Kd * D1);
				if ((I > 0 || only_pd > cMin) && ((I < 0 || only_pd < cMax)))
					I += P * dt;
				input1 = only_pd + Ki * I;
			}
			if (input1 > cMax)input1 = cMax;
			else if (input1 < cMin)input1 = cMin;
			seek = seek_val;
			P = P1;
			D = D1;
			t = t1;
			return input = input1;
		}
	};
}