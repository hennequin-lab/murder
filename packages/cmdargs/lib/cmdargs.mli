(** Simple parsing of command-line arguments.

    The arguments are read once from {!Sys.argv} when the module is loaded. *)

(** Raised when an argument is missing or cannot be parsed. *)
exception Cmdargs_Wrong of string

(** The process arguments, including the program name. *)
val arguments : string list

(** [check s] is [true] when [s] appears in {!arguments}. *)
val check : string -> bool

val get_bool : string -> bool option
val get_int : string -> int option
val get_float : string -> float option

(** [get_* s] parses the value that follows [s], or is [None] when [s] is
    absent. A present argument with no value raises {!Cmdargs_Wrong}. *)
val get_string : string -> string option

(** [force ~usage x] is the value in [x], or raises [Failure usage] when [x] is
    [None]. *)
val force : usage:string -> 'a option -> 'a

(** [default d x] is the value in [x], or [d] when [x] is [None]. *)
val default : 'a -> 'a option -> 'a

(** [in_dir flag name] is [directory/name], where [directory] is the value of
    [flag]. The flag is required. *)
val in_dir : string -> string -> string
