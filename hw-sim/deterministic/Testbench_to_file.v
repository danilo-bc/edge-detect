module Testbench_to_file();
	parameter src_rows = 147;
	parameter src_cols = 143;
	parameter src_size = src_rows*src_cols;
	parameter edge_size = (src_rows-1)*(src_cols-1);
	reg [7:0] src [0:src_size-1];
	reg [7:0] edges [0:edge_size-1];

	// Variable to save file handle
	integer f;
	// Auxiliary variables for counting in loops
	integer i,j;

	// Registers and wires for the Sobel module
	reg [7:0] z1;
	reg [7:0] z2;
	reg [7:0] z3;
	reg [7:0] z4;
	reg [7:0] z6;
	reg [7:0] z7;
	reg [7:0] z8;
	reg [7:0] z9;

	wire [7:0] z_out;

	// Auxiliary system signals
	// May be used in the future
	reg clk;
	reg reset;

	// Instantiate device under test (DUT)
	sobel3x3det DUT(.z1(z1),
					 .z2(z2),
					 .z3(z3),
					 .z4(z4),
					 .z6(z6),
					 .z7(z7),
					 .z8(z8),
					 .z9(z9),
					 .clk(clk),
					 .reset(reset),
					 .z_out(z_out)
					);

	// Clock may be used later for stochastic
	always begin
		#1 clk <= ~clk;
	end

	initial begin
		// Default inputs for unit tests
		clk = 0;
		reset = 0;
		z1 = 8'h00;
		z2 = 8'h00;
		z3 = 8'h00;
		z4 = 8'h00;
		z6 = 8'h00;
		z7 = 8'h00;
		z8 = 8'h00;
		z9 = 8'h00;
		#2
		reset = 1;
		#2
		reset = 0;

		#2
		//Load image coded in hexadecimal as memory
		$dumpfile("test.vcd");
		$dumpvars(0,Testbench_to_file);
		$display("Loading image into memory");
		$readmemh("src.txt",src);
		f = $fopen("edges_hw_det.txt","w");
		// 'Crop' area around central pixel z5
		for (i=1;i<src_rows-1;i=i+1) begin
			for(j=1;j<src_cols-1;j=j+1) begin
				// Row above z5
				z1 <= src[src_cols*(i-1)+j-1];
				z2 <= src[src_cols*(i-1)+j];
				z3 <= src[src_cols*(i-1)+j+1];
				// Row of z5
				z4 <= src[src_cols*(i)+j-1];
				//unused middle pixel
				z6 <= src[src_cols*(i)+j+1];
				// Row below z5
				z7 <= src[src_cols*(i+1)+j-1];
				z8 <= src[src_cols*(i+1)+j];
				z9 <= src[src_cols*(i+1)+j+1];

				#2 // give an instant for z_out to update
				//save 'edges' array for possible future use
				edges[src_cols*(i-1)+j-1]=z_out;
				//Write on file
				$fwrite(f,"%x ",z_out);
			end
		$fwrite(f,"\n");
		end
		$fclose(f);

		$finish;
	end

endmodule