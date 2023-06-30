// RUN: macronify %s | FileCheck %s --match-full-lines

//CHECK: hl.translation_unit {
//CHECK:   hl.typedef "__int128_t" : !hl.int128
//CHECK:   hl.typedef "__uint128_t" : !hl.int128< unsigned >
//CHECK:   hl.struct "__NSConstantString_tag" : {
//CHECK:     hl.field "isa" : !hl.ptr<!hl.int< const >>
//CHECK:     hl.field "flags" : !hl.int
//CHECK:     hl.field "str" : !hl.ptr<!hl.char< const >>
//CHECK:     hl.field "length" : !hl.long
//CHECK:   }
//CHECK:   hl.typedef "__NSConstantString" : !hl.record<"__NSConstantString_tag">
//CHECK:   hl.typedef "__builtin_ms_va_list" : !hl.ptr<!hl.char>
//CHECK:   hl.struct "__va_list_tag" : {
//CHECK:     hl.field "gp_offset" : !hl.int< unsigned >
//CHECK:     hl.field "fp_offset" : !hl.int< unsigned >
//CHECK:     hl.field "overflow_arg_area" : !hl.ptr<!hl.void>
//CHECK:     hl.field "reg_save_area" : !hl.ptr<!hl.void>
//CHECK:   }
//CHECK:   hl.typedef "__builtin_va_list" : !hl.array<1, !hl.record<"__va_list_tag">>
//CHECK:   hl.typedef "size_t" : !hl.long< unsigned >
//CHECK:   hl.struct "contained" : {
//CHECK:   }
//CHECK:   hl.struct "container" : {
//CHECK:     hl.field "contained_member" : !hl.ptr<!hl.elaborated<!hl.record<"contained">>>
//CHECK:   }
//CHECK:   hl.func external @main () -> !hl.int {
//CHECK:     hl.scope {
//CHECK:       %0 = hl.var "container_instance" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"container">>>>
//CHECK:       %1 = macroni.expansion "container_of(ptr, type, member)" : !hl.ptr<!hl.elaborated<!hl.record<"container">>> {
//CHECK:         %3 = hl.stmt.expr : !hl.ptr<!hl.elaborated<!hl.record<"container">>> {
//CHECK:           %4 = hl.var "__mptr" : !hl.lvalue<!hl.ptr<!hl.typeof.expr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>>> = {
//CHECK:             %7 = hl.expr : !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>> {
//CHECK:               %9 = macroni.parameter "ptr" : !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>> {
//CHECK:                 %10 = hl.ref %0 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"container">>>>
//CHECK:                 %11 = hl.implicit_cast %10 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"container">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"container">>>
//CHECK:                 %12 = hl.member %11 at "contained_member" : !hl.ptr<!hl.elaborated<!hl.record<"container">>> -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
//CHECK:                 %13 = hl.addressof %12 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>> -> !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
//CHECK:                 hl.value.yield %13 : !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
//CHECK:               }
//CHECK:               hl.value.yield %9 : !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
//CHECK:             }
//CHECK:             %8 = hl.implicit_cast %7 NoOp : !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>> -> !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
//CHECK:             hl.value.yield %8 : !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
//CHECK:           }
//CHECK:           %5 = hl.expr : !hl.ptr<!hl.char> {
//CHECK:             %7 = hl.ref %4 : !hl.lvalue<!hl.ptr<!hl.typeof.expr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>>>
//CHECK:             %8 = hl.implicit_cast %7 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.typeof.expr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>>> -> !hl.ptr<!hl.typeof.expr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>>
//CHECK:             %9 = hl.cstyle_cast %8 BitCast : !hl.ptr<!hl.typeof.expr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>> -> !hl.ptr<!hl.char>
//CHECK:             %10 = macroni.expansion "offsetof(TYPE, MEMBER)" : !hl.elaborated<!hl.typedef<"size_t">> {
//CHECK:               %12 = hl.expr : !hl.elaborated<!hl.typedef<"size_t">> {
//CHECK:                 %13 = hl.expr : !hl.ptr<!hl.elaborated<!hl.record<"container">>> {
//CHECK:                   %17 = hl.const #hl.integer<0> : !hl.int
//CHECK:                   %18 = hl.implicit_cast %17 NullToPointer : !hl.int -> !hl.ptr<!hl.elaborated<!hl.record<"container">>>
//CHECK:                   %19 = hl.cstyle_cast %18 NoOp : !hl.ptr<!hl.elaborated<!hl.record<"container">>> -> !hl.ptr<!hl.elaborated<!hl.record<"container">>>
//CHECK:                   hl.value.yield %19 : !hl.ptr<!hl.elaborated<!hl.record<"container">>>
//CHECK:                 }
//CHECK:                 %14 = hl.member %13 at "contained_member" : !hl.ptr<!hl.elaborated<!hl.record<"container">>> -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
//CHECK:                 %15 = hl.addressof %14 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>> -> !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
//CHECK:                 %16 = hl.cstyle_cast %15 PointerToIntegral : !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>> -> !hl.elaborated<!hl.typedef<"size_t">>
//CHECK:                 hl.value.yield %16 : !hl.elaborated<!hl.typedef<"size_t">>
//CHECK:               }
//CHECK:               hl.value.yield %12 : !hl.elaborated<!hl.typedef<"size_t">>
//CHECK:             }
//CHECK:             %11 = hl.sub %9, %10 : (!hl.ptr<!hl.char>, !hl.elaborated<!hl.typedef<"size_t">>) -> !hl.ptr<!hl.char>
//CHECK:             hl.value.yield %11 : !hl.ptr<!hl.char>
//CHECK:           }
//CHECK:           %6 = hl.cstyle_cast %5 BitCast : !hl.ptr<!hl.char> -> !hl.ptr<!hl.elaborated<!hl.record<"container">>>
//CHECK:           hl.value.yield %6 : !hl.ptr<!hl.elaborated<!hl.record<"container">>>
//CHECK:         }
//CHECK:         hl.value.yield %3 : !hl.ptr<!hl.elaborated<!hl.record<"container">>>
//CHECK:       }
//CHECK:       %2 = hl.const #hl.integer<0> : !hl.int
//CHECK:       hl.return %2 : !hl.int
//CHECK:     }
//CHECK:   }
//CHECK: }

typedef unsigned long size_t;

#define offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)

#define container_of(ptr, type, member) ({          \
	const typeof(((type *)0)->member)*__mptr = (ptr);    \
		     (type *)((char *)__mptr - offsetof(type, member)); })

struct contained {};

struct container {
        struct contained *contained_member;
};

int main(void) {
        struct container *container_instance;
        container_of(&container_instance->contained_member, struct container, contained_member);
        return 0;
}
