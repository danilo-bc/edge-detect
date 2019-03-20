module Testbench_to_file();
	//parameter src_rows = 436;
	//parameter src_cols = 576;
	parameter src_rows = 3;
	parameter src_cols = 4;
	parameter src_size = src_rows*src_cols;
	//parameter edge_size = (src_rows-1)*(src_cols-1);
	reg [7:0] src [0:src_size-1];
	//reg [11:0] edges [0:edge_size-1];

	// Variable to save file handle
	integer f;
	// Auxiliary variables for counting in loops
	integer i,j;

	reg [7:0] z1;
	reg [7:0] z2;
	reg [7:0] z3;
	reg [7:0] z4;
	reg [7:0] z5;
	reg [7:0] z6;

	wire [11:0] z_out; //binary value of output

	reg clk;
	reg reset;

	sobel3x3det det1(.z1(z1),
					 .z2(z2),
					 .z3(z3),
					 .z4(z4),
					 .z5(z5),
					 .z6(z6),
					 .z_out(z_out)
					);

	always begin
		#1 clk <= ~clk;
	end

	initial begin
		z1 = 8'h01;
		z2 = 8'h00;
		z3 = 8'h00;
		z4 = 8'h00;
		z5 = 8'h00;
		z6 = 8'h00;

		#2
		//Load image coded in hexadecimal as memory
		$display("z_out = %x",z_out);

		$display("Loading image into memory");
		$readmemh("teste.txt",src);
		f = $fopen("teste_after_load.txt","w");
		for (i=1;i<src_rows-1;i=i+1)
		begin
			for(j=1;j<src_cols-1;j=j+1)
			begin
				// Input for Gy
				z1 = src[src_cols*(i-1)+j-1];
				z2 = src[src_cols*(i-1)+j];
				z3 = src[src_cols*(i-1)+j+1];
				z4 = src[src_cols*(i+1)+j-1];
				z5 = src[src_cols*(i+1)+j];
				z6 = src[src_cols*(i+1)+j+1];

				#2 // give an instant for z_out to update
				//edges[src_cols*(i-1)+j-1]=z_out;
				//$display("edges: %h",edges[src_cols*(i-1)+j-1]);
				$fwrite(f,"%x ",z_out);
			end
			$fwrite(f,"\n");
		end
		$fclose(f);

		//$display("Loading reference image into memory");
		//$readmemh("edges.txt",edges);
//TODO: automatically compare edges.txt and output of test up until now

		//$stop;
		$finish;
	end

endmodule