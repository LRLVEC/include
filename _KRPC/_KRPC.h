#pragma once
#include <krpc.hpp>
#include <krpc/services/krpc.hpp>
#include <krpc/services/space_center.hpp>

#ifdef _DEBUG
#pragma comment(lib,"debug/krpc.lib")
#pragma comment(lib,"debug/libprotobufd.lib")
#else
#pragma comment(lib,"release/krpc.lib")
#pragma comment(lib,"release/libprotobuf.lib")
#endif

namespace KRPC
{

}