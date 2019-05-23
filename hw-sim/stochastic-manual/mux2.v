module mux2(input i1,
			input i2,
			input s1,
			output reg out
			);

always @(*) begin
	case(s1)
		1'b0: out <= i1;
		1'b1: out <= i2;
	endcase
end
endmodule
