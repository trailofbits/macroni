// RUN: macronify -xc %s --convert | FileCheck %s --match-full-lines

// CHECK: hl.translation_unit {
// CHECK:   hl.typedef "__int128_t" : !hl.int128
// CHECK:   hl.typedef "__uint128_t" : !hl.int128< unsigned >
// CHECK:   hl.struct "__NSConstantString_tag" : {
// CHECK:     hl.field "isa" : !hl.ptr<!hl.int< const >>
// CHECK:     hl.field "flags" : !hl.int
// CHECK:     hl.field "str" : !hl.ptr<!hl.char< const >>
// CHECK:     hl.field "length" : !hl.long
// CHECK:   }
// CHECK:   hl.typedef "__NSConstantString" : !hl.record<"__NSConstantString_tag">
// CHECK:   hl.typedef "__builtin_ms_va_list" : !hl.ptr<!hl.char>
// CHECK:   hl.struct "__va_list_tag" : {
// CHECK:     hl.field "gp_offset" : !hl.int< unsigned >
// CHECK:     hl.field "fp_offset" : !hl.int< unsigned >
// CHECK:     hl.field "overflow_arg_area" : !hl.ptr<!hl.void>
// CHECK:     hl.field "reg_save_area" : !hl.ptr<!hl.void>
// CHECK:   }
// CHECK:   hl.typedef "__builtin_va_list" : !hl.array<1, !hl.record<"__va_list_tag">>
// CHECK:   hl.typedef "size_t" : !hl.long< unsigned >
// CHECK:   hl.struct "contained" : {
// CHECK:   }
// CHECK:   hl.struct "container" : {
// CHECK:     hl.field "contained_member" : !hl.ptr<!hl.elaborated<!hl.record<"contained">>>
// CHECK:   }
// CHECK:   hl.func external @main () -> !hl.int {
// CHECK:     hl.scope {
// CHECK:       %0 = hl.var "container_instance" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"container">>>>
// CHECK:       %1 = macroni.parameter "ptr" : !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>> {
// CHECK:         %4 = hl.ref %0 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"container">>>>
// CHECK:         %5 = hl.implicit_cast %4 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"container">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"container">>>
// CHECK:         %6 = hl.member %5 at "contained_member" : !hl.ptr<!hl.elaborated<!hl.record<"container">>> -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
// CHECK:         %7 = hl.addressof %6 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>> -> !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
// CHECK:         hl.value.yield %7 : !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
// CHECK:       }
// CHECK:       %2 = kernel.container_of container_of(%1, !hl.elaborated<!hl.record<"container">>, "contained_member") : (!hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>) -> !hl.ptr<!hl.elaborated<!hl.record<"container">>>
// CHECK:       %3 = hl.const #hl.integer<0> : !hl.int
// CHECK:       hl.return %3 : !hl.int
// CHECK:     }
// CHECK:   }
// CHECK: }

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
