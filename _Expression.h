#pragma once
#include <cstdlib>
#include <new>

namespace Expression
{
	enum NodeType
	{
		IsOprand,
		IsOperator
	};
	template<class T>struct Oprand
	{
		static constexpr const NodeType type = IsOprand;
		virtual T operator()()const = 0;
		virtual T operator=(T const&) = 0;
	};
	template<class T>struct Operator
	{
		static constexpr const NodeType type = IsOperator;
		virtual T operator()(Oprand<T>const*, Oprand<T>const*)const = 0;
	};

	// All memories are controlled inside.
	// R must has default construct function.
	template<class T, class R, class S>struct Node
	{
		Oprand<T>* oprd;
		Operator<T>* oprt;
		Node<T, R, S>* left, * right;
		static_assert(R::type == IsOprand, "R is not a correct Oprand<> struct!");
		static_assert(S::type == IsOperator, "S is not a correct Operator<> struct!");
		Node(T const& a)
			:
			oprd((R*)::malloc(sizeof(R))),
			oprt(nullptr),
			left(nullptr),
			right(nullptr)
		{
			new(oprd)R(a);
		}
		Node(T* a)
			:
			oprd((R*)::malloc(sizeof(R))),
			oprt(nullptr),
			left(nullptr),
			right(nullptr)
		{
			new(oprd)R(a);
		}
		template<class M>Node(M const& a)
			:
			oprd((R*)::malloc(sizeof(R))),
			oprt((S*)::malloc(sizeof(S))),
			left(nullptr),
			right(nullptr)
		{
			new(oprd)R;
			new(oprt)S(a);
		}
		~Node()
		{
			::free(oprd);
			::free(oprt);
			oprd = nullptr;
			oprt = nullptr;
			if (left) { left->~Node(); left = nullptr; }
			if (right) { right->~Node(); left = nullptr; }
		}
		Oprand<T> const& operator()()
		{
			if (left || right)(*oprd) = (*oprt)(left ? &(*left)() : nullptr, right ? &(*right)() : nullptr);
			return *oprd;
		}
	};
}