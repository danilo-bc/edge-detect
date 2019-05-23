all: Testbench_to_file.v
	iverilog -y . -o simul.iver Testbench_to_file.v
	vvp simul.iver

clean:
	rm *.iver
