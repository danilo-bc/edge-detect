module abs_diff(input [11:0] oper1,
				input [11:0] oper2,
				output reg [11:0] result
				);

	always @(*) begin
		if (oper1>oper2) begin
			result = (oper1-oper2);
		end
		else begin
			result = (oper2-oper1);
		end

	end

endmodule