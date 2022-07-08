# import math
# x = 500

# def toBinary(a):
#   l,m=[],[]
#   for i in a:
#     l.append(ord(i))
#   for i in l:
#     m.append(int(bin(i)[2:]))
#   return m

# print(x) 
# print(toBinary(str(x)))
# -----------------------------------------------------


# # Create bytearray
# a_string = "500"
# a_byte_array = bytearray(a_string, "utf8")
# byte_list = []

# # Convert to binary Add to list
# for byte in a_byte_array:
#     binary_representation = bin(byte)

#     byte_list.append(binary_representation)



# print(byte_list)
# --------------------------------------------------------

temp = format(500, "b")

print(temp)