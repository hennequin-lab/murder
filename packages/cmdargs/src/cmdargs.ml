(** Simpler parsing of the command-line *)

(* backup the command line *)
let arguments = Array.to_list Sys.argv

exception Cmdargs_Wrong of string

(** Check for the presence of a specific argument in the command line *)
let check s = List.mem s arguments

let extract f x err =
  try f x with
  | _ -> raise (Cmdargs_Wrong err)


let _get f err s =
  let rec search = function
    | [] -> None
    | o :: rest ->
      if o = s
      then (
        match rest with
        | [] -> raise (Cmdargs_Wrong err)
        | x :: _ -> Some (extract f x err))
      else search rest
  in
  search arguments


let get_bool = _get bool_of_string "get_bool"
let get_int = _get int_of_string "get_int"
let get_float = _get float_of_string "get_float"
let get_string = _get (fun s -> (s : string)) "get_string"

let force ~usage x =
  match x with
  | None -> failwith usage
  | Some z -> z


let default d x =
  match x with
  | None -> d
  | Some z -> z


let in_dir cmdarg =
    let usage = Printf.sprintf "%s [directory]" cmdarg in
    let dir = get_string cmdarg |> force ~usage in
    Printf.sprintf "%s/%s" dir
