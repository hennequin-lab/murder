
lib:
	ocamlbuild cmdargs.cma
	ocamlbuild cmdargs.cmxa

install:
	@ocamlfind remove cmdargs
	@ocamlfind install cmdargs META _build/cmdargs.cmi \
		_build/cmdargs.cma _build/cmdargs.cmxa _build/cmdargs.a _build/cmdargs.cmx

all: lib install

clean:
	ocamlbuild -clean
	
