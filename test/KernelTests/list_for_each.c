// RUN: kernelize -x c %s | FileCheck %s --match-full-lines

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
// CHECK:   hl.struct "list_head" : {
// CHECK:     hl.field "next" : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
// CHECK:     hl.field "prev" : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
// CHECK:   }
// CHECK:   hl.func internal @list_is_head (%arg0: !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>>, %arg1: !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>>) -> !hl.int attributes {sym_visibility = "private"} {
// CHECK:     hl.scope {
// CHECK:       %0 = hl.ref %arg0 : (!hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>>) -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>>
// CHECK:       %1 = hl.implicit_cast %0 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>
// CHECK:       %2 = hl.ref %arg1 : (!hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>>) -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>>
// CHECK:       %3 = hl.implicit_cast %2 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>
// CHECK:       %4 = hl.cmp eq %1, %3 : !hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>, !hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >> -> !hl.int
// CHECK:       hl.return %4 : !hl.int
// CHECK:     }
// CHECK:   }
// CHECK:   hl.func external @main () -> !hl.int {
// CHECK:     hl.scope {
// CHECK:       %0 = hl.var "head" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
// CHECK:       %1 = hl.var "pos" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
// CHECK:       hl.scope {
// CHECK:         %3 = macroni.parameter "pos" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
// CHECK:           %11 = hl.ref %1 : (!hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>) -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
// CHECK:           hl.value.yield %11 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
// CHECK:         }
// CHECK:         %4 = hl.expr : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
// CHECK:           %11 = macroni.parameter "head" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
// CHECK:             %12 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>) -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
// CHECK:             hl.value.yield %12 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
// CHECK:           }
// CHECK:           hl.value.yield %11 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
// CHECK:         }
// CHECK:         %5 = hl.implicit_cast %4 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
// CHECK:         %6 = hl.member %5 at "next" : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>> -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
// CHECK:         %7 = hl.implicit_cast %6 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
// CHECK:         %8 = hl.assign %7 to %3 : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>, !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
// CHECK:         %9 = macroni.parameter "pos" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
// CHECK:           %11 = hl.ref %1 : (!hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>) -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
// CHECK:           hl.value.yield %11 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
// CHECK:         }
// CHECK:         %10 = macroni.parameter "head" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
// CHECK:           %11 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>) -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
// CHECK:           hl.value.yield %11 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
// CHECK:         }
// CHECK:         kernel.list_for_each() list_for_each(%9, %10) : (!hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>, !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>) -> () {
// CHECK:           hl.scope {
// CHECK:             %11 = hl.var "prev" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> = {
// CHECK:               %12 = hl.ref %1 : (!hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>) -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
// CHECK:               %13 = hl.implicit_cast %12 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
// CHECK:               %14 = hl.member %13 at "prev" : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>> -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
// CHECK:               %15 = hl.implicit_cast %14 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
// CHECK:               hl.value.yield %15 : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:       %2 = hl.const #hl.integer<0> : !hl.int
// CHECK:       hl.return %2 : !hl.int
// CHECK:     }
// CHECK:   }
// CHECK: }

typedef unsigned long size_t;

struct list_head { struct list_head *next, *prev; };

static inline int list_is_head(const struct list_head *list,
			       const struct list_head *head) {
	return list == head;
}

#define list_for_each(pos, head) \
	for (pos = (head)->next; !list_is_head(pos, (head)); pos = pos->next)

int main(void) {
	struct list_head *head, *pos;
	list_for_each(pos, head) {
		struct list_head *prev = pos->prev;
	}
	return 0;
}
