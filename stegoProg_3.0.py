from PIL import Image
import os
from customtkinter import *
import PIL.ImageChops
import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt
from math import log10, sqrt 
from timeit import default_timer as timer
import itertools

quant = np.array([[16,11,10,16,24,40,51,61],      # QUANTIZATION TABLE
                    [12,12,14,19,26,58,60,55],    # required for DCT
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])


def textToBinary(text):
    binary_message = ''.join(format(ord(char), '08b') for char in text)
    return binary_message

# for char in text: Цикл проходить крізь кожен символ у тексті.
# format(ord(char), '08b'): Для кожного символу використовується функція ord, щоб перетворити його в його ASCII-код, а потім за допомогою format перетворюється цей код в бінарний рядок, де '08b' вказує, що потрібно використовувати 8 біт для представлення числа, і вивід буде доповнений нулями зліва при необхідності.
# binary_message = ''.join(...): Отримані бінарні рядки для кожного символу об'єднуються разом у єдиний рядок binary_message.

def binaryToText(binary_str):
    text = ''.join(chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8))
    return text
    
def addPadd(img, row, col):
        img = cv2.resize(img,(col+(8-col%8),row+(8-row%8)))    
        return img
    
def chunks(l, n):    
        m = int(n)
        for i in range(0, len(l), m):
            yield l[i:i + m]

def textToUnicode(message):
    bits = []
    for char in message:
        binval = bin(ord(char))[2:].rjust(8,'0')
        bits.append(binval)
        # print("bits", bits)
    return bits

def browseFile_to_hide(file_entry,result_label, props_label):
    try:
        file_path = filedialog.askopenfilename(title="Select A File", filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("BMP files", "*.bmp"), ("All files", "*.*")))
        
        file_entry.delete(0, file_entry.index(END))
        file_entry.insert(0, file_path)
        
        
        display_image_preview_before(file_path)
        # getFileProperties(file_path,result_label, format_label, size_label, width_label, height_label, bit_depth_label)
        
        
    except Exception as e:
        # result_label.configure(text=f"Error: {str(e)}")
        result_label.configure(text="Select a file!")
        props_label.configure(text="")
    
def browseFile_to_reveal(file_entry,result_label, props_label, text_area):
    try:
        file_path = filedialog.askopenfilename(title="Select A File", filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("BMP files", "*.bmp"), ("All files", "*.*")))
        
        file_entry.delete(0, file_entry.index(END))
        file_entry.insert(0, file_path)
        
        
        display_image_preview_after(file_path)
        # getFileProperties(file_path,result_label, format_label, size_label, width_label, height_label, bit_depth_label)
        
        
    except Exception as e:
        # result_label.configure(text=f"Error: {str(e)}")     
        result_label.configure(text="Select a file!")    
        props_label.configure(text="")
        text_area.configure(state='normal')
        text_area.delete("1.0", "end")
        text_area.configure(state='disabled')
        
def browseFile_to_compare(file_entry,result_label):
    try:
        file_path = filedialog.askopenfilename(title="Select A File", filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("BMP files", "*.bmp"), ("All files", "*.*")))
        
        file_entry.delete(0, file_entry.index(END))
        file_entry.insert(0, file_path)
        
    except Exception as e:
        # result_label.configure(text=f"Error: {str(e)}")    
        result_label.configure(text="Select a file!")

def embed_text(file_path_entry, text_to_hide_entry, result_label, props_label):
    if hide_text_window_option_method.get() == "Least Significant Bit":
        hideText_LSB(file_path_entry, text_to_hide_entry,result_label, props_label)
    elif hide_text_window_option_method.get() == "Discrete Cosine Transform":
        hideText_DCT(file_path_entry, text_to_hide_entry,result_label, props_label)
        
        
def extract_text(file_path_entry, text_to_hide_entry,result_label, props_label):
    if reveal_text_window_option_method.get() == "Least Significant Bit":
        revealText_LSB(file_path_entry, text_to_hide_entry,result_label, props_label)
    elif reveal_text_window_option_method.get() == "Discrete Cosine Transform":
        revealText_DCT(file_path_entry, text_to_hide_entry,result_label, props_label)
        
        
def hideText_LSB(file_path_entry, text_to_hide_entry,result_label, props_label):
    try:
        start = timer()
        end_marker = "###END###"
        # text_to_hide = text_to_hide_entry.get(1.0, "end-1c")
        message_to_hide = text_to_hide_entry.get(1.0, "end-1c")
        file_path = file_path_entry.get()
        
        
        
        
        if os.path.isfile(file_path):
            if message_to_hide.strip() != "":
                
                message_to_hide = text_to_hide_entry.get(1.0, "end-1c") + end_marker
                # Конвертуємо кириличний текст у бінарний формат
                # text_to_hide_bytes = text_to_hide.encode('utf-8')
                # binary_text = ''.join(format(byte, '08b') for byte in text_to_hide_bytes) + textToBinary(end_marker)
            
                image = PIL.Image.open(file_path)
                width, height = image.size
                img_arr = np.array(list(image.getdata()))
                
                
               
                    
                if image.mode == "P":
                    result_label.configure(text="Not supported!")
                    
                channels = 4 if image.mode =="RGBA" else 3
                
                pixels = img_arr.size // channels
                
                byte_message = ''.join(f"{ord(c):08b}" for c in message_to_hide)
                bits = len(byte_message)
                
                
                if bits > pixels:
                    result_label.configure(text="The message is too long!")
                else:
                    index = 0
                    for i in range(pixels):
                        for j in range(0, 3):
                            if index < bits:
                                img_arr[i][j] = int(bin(img_arr[i][j])[2:-1] + byte_message[index], 2)
                                index += 1
                                
                                
                img_arr = img_arr.reshape((height, width, channels))
                result = PIL.Image.fromarray(img_arr.astype('uint8'), image.mode)
                
                result_path = './imgs/encoded/encoded_image_lsb.png'
                display_image_preview_after(result_path)
                
                result.save(result_path)              
                                
                                
                # img = img.convert('RGB')
                # pixels = list(img.getdata())
                # width, height = img.size
                
                # if len(binary_text) > len(pixels) * 3:
                #     result_label.configure(text="Text is too long to be hidden in the image.")

                # new_pixels = []
                # index = 0

                # for pixel_value in pixels:
                #     pixel_value_bin = list(pixel_value)
                #     for i in range(3):
                #         if index < len(binary_text):
                #             pixel_value_bin[i] = int(format(pixel_value_bin[i], '08b')[:-1] + binary_text[index], 2)
                #             index += 1
                #     new_pixels.append(tuple(pixel_value_bin))

                # new_image = Image.new('RGB', (width, height))
                # new_image.putdata(new_pixels)
                # new_image.save('hidden_image.png')
                end = timer()
                
                print(file_path, "        ", result_path)
                
                processing_time = round(end - start, 3)
                
                psnr_mse(file_path, result_path, props_label, processing_time)
                
                
                print(str(start) + "       "+ str(end)+"           " + str(end-start))
                result_label.configure(text="Text hidden successfully!") 
            else: 
                result_label.configure(text="Type a text you want to hide!")
                props_label.configure(text="")
        else:
            result_label.configure(text="No such file or directory!") 
            props_label.configure(text="")
            
    except Exception as e:
            # result_label.configure(text=f"Error: {str(e)}")
            print(f"Error: {str(e)}")
            props_label.configure(text="")
            
            result_label.configure(text="No such file or directory!") 

def revealText_LSB(file_path_entry,  result_text,  result_label, time_label):
    try:
        start = timer()
        
        file_path = file_path_entry.get()
        end_marker = "###END###"
        if os.path.isfile(file_path):
            image = PIL.Image.open(file_path)
            img_arr = np.array(list(image.getdata()))
            
            # img = Image.open(file_path)
            # pixels = list(img.getdata())
            # binary_text = ''
            
            channels = 4 if image.mode == 'RGBA' else 3
            
            pixels = img_arr.size // channels
            
            secret_bits = [bin(img_arr[i][j])[-1] for i in range(pixels) for j in range(0,3)]
            secret_bits = ''.join(secret_bits)
            secret_bits = [secret_bits[i:i+8] for i in range(0, len(secret_bits), 8)]

            secret_message = [chr(int(secret_bits[i], 2)) for i in range(len(secret_bits))]
            secret_message = ''.join(secret_message)
            
            if end_marker in secret_message:
                result_text.configure(state='normal')
                result_text.delete('1.0', END)
                
                result_text.insert(INSERT, secret_message[:secret_message.index(end_marker)])
                result_text.configure(state='disabled')
                
            else:
                result_label.configure(text="Couldn't find secret message!")
            
            end = timer()
            
            # for pixel_value in pixels:
            #     for i in range(3):
            #         binary_text += format(pixel_value[i], '08b')[-1]

            # Assume '###END###' as the end marker
            # end_index = binary_text.find(textToBinary("###END###"))
            # if end_index != -1:
            #     binary_text = binary_text[:end_index]

            # # Decode binary text to bytes
            # binary_bytes = [binary_text[i:i+8] for i in range(0, len(binary_text), 8)]
            # binary_bytes = [int(byte, 2) for byte in binary_bytes]

            # # Convert bytes to text
            # decrypted_text = bytes(binary_bytes).decode('utf-8')
            # result_label.configure(text=f"Decrypted Text: {decrypted_text}")
            
            time_label.configure(text="Extraction time = " + str(round(end-start, 3)))
            
        else:
            result_label.configure(text="No such file or directory!")
            
    except Exception as e:
        result_label.configure(text=f"Error: {str(e)}")


def hideText_DCT(file_path_entry, text_to_hide_entry, result_label, props_label):
    try:  
        start = timer()
        file_path = file_path_entry.get()
        
        secret_message = text_to_hide_entry.get(1.0, "end-1c")
        
        
        if os.path.isfile(file_path_entry.get()):
            if secret_message.strip() != "":
                message = str(len(secret_message)) + '*' + secret_message
                bit_message = textToUnicode(message)
                # print(bit_message)
                
                
                
                img = cv2.imread(file_path)
    
                
                #get size of image in pixels
                row,col = img.shape[:2]
                if((col/8)*(row/8)<len(secret_message)):
                    result_label.configure("The message is too long!")
                    print("The message is too long!")
                
                
                #make divisible by 8x8
                if row%8 != 0 or col%8 != 0:
                    img = addPadd(img, row, col)
                
                row,col = img.shape[:2]
                ##col, row = img.size
                #split image into RGB channels
                img_blue, img_green, img_red = cv2.split(img)
                #message to be hid in blue channel so converted to type float32 for dct function
                img_blue = np.float32(img_blue)
                #break into 8x8 blocks
                img_blocks = [np.round(img_blue[j:j+8, i:i+8]-128) for (j,i) in itertools.product(range(0,row,8), range(0,col,8))]
                #Blocks are run through DCT function
                dct_blocks = [np.round(cv2.dct(img_Block)) for img_Block in img_blocks]
                #blocks then run through quantization table
                quantized_dct = [np.round(dct_Block/quant) for dct_Block in dct_blocks]
                #set LSB in DC value corresponding bit of message
                message_index = 0
                letter_index = 0
                
                for quantized_block in quantized_dct:
                    #find LSB in DC coeff and replace with message bit
                    # print("quantized block = ", quantized_block)
                    DC = quantized_block[0][0]
                    # print("dc =", DC)
                    DC = np.uint8(DC)
                    DC = np.unpackbits(DC)
                    # print("dc in bits = ", DC)
                    # print("DC ",DC[len(DC)-1], "замін.ється на ", bit_message[message_index][letter_index])
                    DC[len(DC)-1] = bit_message[message_index][letter_index]
                    DC = np.packbits(DC)
                    DC = np.float32(DC)
                    DC= DC-255
                    quantized_block[0][0] = DC
                    letter_index += 1
                    if letter_index == 8:
                        letter_index = 0
                        message_index += 1
                    
                    if message_index == len(message):
                        break
                        
                #blocks run inversely through quantization table
                sImgBlocks = [quantized_block *quant+128 for quantized_block in quantized_dct]
                #blocks run through inverse DCT
                #puts the new image back together
                sImg=[]
                for chunkRowBlocks in chunks(sImgBlocks, col/8):
                    for rowBlockNum in range(8):
                        for block in chunkRowBlocks:
                            sImg.extend(block[rowBlockNum])
                sImg = np.array(sImg).reshape(row, col)
                #converted from type float32
                sImg = np.uint8(sImg)
                #show(sImg)
                
                
                sImg = cv2.merge((sImg,img_green, img_red))
                end = timer()
                result_path = "./imgs/encoded/encoded_image_dct.png"      
                
                
                           
                cv2.imwrite(result_path, sImg)
                processing_time = round(end-start, 3)
                psnr_mse(file_path, result_path, props_label, processing_time)
                result_label.configure(text="Text hidden successfully!") 
                
                display_image_preview_after(result_path)
                
            else:
                result_label.configure(text="Type a text you want to hide!")
                props_label.configure(text="")
        else:
            result_label.configure(text="No such file or directory!") 
            props_label.configure(text="")
    except Exception as e:
        print(e)
        # result_label.configure(text=f"Error: {str(e)}")
        result_label.configure(text="No such file or directory!") 
        
def revealText_DCT(file_path_entry,  result_text,  result_label, time_label):
    try:
        start = timer()
        img = cv2.imread(file_path_entry.get())
        row, col = img.shape[:2]
        messSize = None
        messageBits = []
        buff = 0
        secret_message= ""

        bImg, gImg, rImg = cv2.split(img)
        bImg = np.float32(bImg)
        gImg = np.float32(gImg)
        rImg = np.float32(rImg)

        imgBlocks = [bImg[j:j+8, i:i+8]-128 for (j,i) in itertools.product(range(0,row,8),
                                                                               range(0,col,8))]    

        quantizedDCT = [img_Block / quant for img_Block in imgBlocks]
        i = 0

        for quantizedBlock in quantizedDCT:
            DC = quantizedBlock[0][0]
            DC = np.uint8(DC)
            DC = np.unpackbits(DC)
            # print('dc =', DC)
            if DC[7] == 1:
                buff += (0 & 1) << (7 - i)
            elif DC[7] == 0:
                buff += (1 & 1) << (7 - i)

            i += 1

            if i == 8:
                messageBits.append(chr(buff))
                buff = 0
                i = 0
                # print(messageBits[-1])
                if messageBits[-1] == '*' and messSize is None:
                    try:
                        messSize = int(''.join(messageBits[:-1]))
                        # print(messSize)
                    except:
                        pass
                    
            if len(messageBits) - len(str(messSize)) - 1 == messSize:
                secret_message = ''.join(messageBits)[len(str(messSize)) + 1:]
            
        sImgBlocks = [quantizedBlock * quant + 128 for quantizedBlock in quantizedDCT]

        sImg = []
        for chunkRowBlocks in chunks(sImgBlocks, col / 8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])

        sImg = np.array(sImg).reshape(row, col)
        sImg = np.float32(sImg)

        # Ensure dimensions match for merge
        gImg = cv2.resize(gImg, (col, row))
        rImg = cv2.resize(rImg, (col, row))

        # print(sImg.dtype)
        # print(gImg.dtype)
        # print(rImg.dtype)
        
        sImg = cv2.merge((sImg, gImg, rImg))

        end = timer()
        
        result_text.configure(state='normal')
        result_text.delete('1.0', END)
                
        result_text.insert(INSERT, secret_message)
        result_text.configure(state='disabled')
        time_label.configure(text="Extraction time = " + str(round(end-start, 3)))
    except Exception as e:
        print(e)
        result_label.configure(text=f"Error: {str(e)}")

  
main_window = CTk()
main_window.title("StegoProg")
main_window.geometry("740x740")
set_appearance_mode("dark")
    
    
tabView = CTkTabview(master=main_window,
                     width=720,
                     height=640)    
tabView.pack(padx=10, pady=10)

hide_tab = tabView.add("Hide Text")   
reveal_tab = tabView.add("Reveal Text")   
preview_tab = tabView.add("Image Preview")  
comparison_tab = tabView.add("Comparison")  
manual_tab = tabView.add("Manual")  


# top_frame = customtkinter.CTkFrame(tabView.tab("Hide Text"))  
# top_frame.pack(side="top")

# top_left_frame = customtkinter.CTkFrame(top_frame)
# top_left_frame.pack(side="left")

# top_rigth_frame = customtkinter.CTkFrame(top_frame)
# top_rigth_frame.pack(side="right")

# image_properties = CTkLabel(top_rigth_frame, text="Image Properties:")
# image_properties.pack()  
# image_format = CTkLabel(top_rigth_frame)
# image_format.pack()
# image_size = CTkLabel(top_rigth_frame)
# image_size.pack()
# image_width = CTkLabel(top_rigth_frame)
# image_width.pack()
# image_height = CTkLabel(top_rigth_frame)
# image_height.pack()
# image_bit_depth = CTkLabel(top_rigth_frame)
# image_bit_depth.pack()    
    
    
    
# --------------------------------------------------------------------------------------
    
hide_text_window_file_label = CTkLabel(hide_tab, text="Select File:")
hide_text_window_file_label.pack(pady=5)
    
hide_text_window_file_entry = CTkEntry(hide_tab,height=20, width=350)
hide_text_window_file_entry.pack(pady=5)

hide_text_window_browse_button = CTkButton(hide_tab,text="Browse",cursor="hand2", command=lambda: browseFile_to_hide(hide_text_window_file_entry, hide_text_window_result_label, hide_text_window_props_label))
hide_text_window_browse_button.pack(pady=5)
    
hide_text_window_text_area = CTkTextbox(hide_tab,height=200,width=400)
hide_text_window_text_area.pack(pady=5)
    
hide_text_window_result_label = CTkLabel(hide_tab,text="")
hide_text_window_result_label.pack(pady=5)

hide_text_window_props_label = CTkLabel(hide_tab,text="")
hide_text_window_props_label.pack(pady=5)
    
hide_text_window_option_method = CTkOptionMenu(hide_tab, values=["Least Significant Bit", "Discrete Cosine Transform"])    
hide_text_window_option_method.pack(pady=20)
    
hide_text_window_hide_button = CTkButton(hide_tab,text="Hide", command= lambda: embed_text(hide_text_window_file_entry, hide_text_window_text_area, hide_text_window_result_label, hide_text_window_props_label), cursor="hand2")
hide_text_window_hide_button.pack(pady=5)

# -----------------------------------------------------------------------------------


reveal_text_window_file_label = CTkLabel(reveal_tab, text="Select File:")
reveal_text_window_file_label.pack(pady=5)
    
reveal_text_window_file_entry = CTkEntry(reveal_tab,height=20, width=350)
reveal_text_window_file_entry.pack(pady=5)

reveal_text_window_browse_button = CTkButton(reveal_tab,text="Browse",cursor="hand2", command=lambda: browseFile_to_reveal(reveal_text_window_file_entry, reveal_text_window_result_label, reveal_text_window_props_label, reveal_text_window_result_text))
reveal_text_window_browse_button.pack(pady=5)



reveal_text_window_result_text = CTkTextbox(reveal_tab,height=200,width=400)
reveal_text_window_result_text.pack(pady=5)
reveal_text_window_result_text.configure(state='disabled')

reveal_text_window_result_label = CTkLabel(reveal_tab,text="")
reveal_text_window_result_label.pack(pady=5)

reveal_text_window_props_label  = CTkLabel(reveal_tab,text="")
reveal_text_window_props_label.pack(pady=5)


reveal_text_window_option_method = CTkOptionMenu(reveal_tab, values=["Least Significant Bit", "Discrete Cosine Transform"])    
reveal_text_window_option_method.pack(pady=20)

reveal_text_window_reveal_button = CTkButton(reveal_tab,text="Reveal Text", command= lambda: extract_text(reveal_text_window_file_entry, reveal_text_window_result_text,reveal_text_window_result_label, reveal_text_window_props_label), cursor="hand2")
reveal_text_window_reveal_button.pack(pady=5)


# ---------------------------------------------------------------------------------------

img_label_before = CTkLabel(preview_tab, text="Before")
img_label_before.pack(pady=10)

img_label_after = CTkLabel(preview_tab, text="After")
img_label_after.pack(pady=10)


# Функція для відображення попереднього перегляду зображення
def display_image_preview_before(image_path):
    img = CTkImage(light_image=Image.open(image_path), dark_image=Image.open(image_path), size=(300, 260))
    img_label_before.configure(image=img)

def display_image_preview_after(image_path):
    img = CTkImage(light_image=Image.open(image_path), dark_image=Image.open(image_path), size=(300, 260))
    img_label_after.configure(image=img)

# Функція для оновлення попереднього перегляду зображення та властивостей
def update_preview():
    update_button = CTkButton(preview_tab, text="Update", command=update_preview)
    update_button.pack()

# ---------------------------------------------------------------------------------------



def compare_red_channel(image1_entry, image2_entry, result_label):
    try: 
        img1 = cv2.imread(image1_entry.get())   
        img2 = cv2.imread(image2_entry.get())

        # Convert images from BGR to RGB
        image1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        image2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # Split images into channels
        _, _, r1 = cv2.split(image1_rgb)
        _, _, r2= cv2.split(image2_rgb)
        
        
        hist_r1 = cv2.calcHist([r1], [0], None, [256], [0,256])
        hist_r2 = cv2.calcHist([r2], [0], None, [256], [0,256])
        plt.figure(figsize=(10, 5))
        # Plot red channel and comparison
        plt.plot(hist_r1, color='red', label='Before')
        plt.plot(hist_r2, color='orange', label='After')
        plt.title('Red Channel')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.xlim([0, 256])
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        result_label.configure(text="Select a File!")
        
    
def compare_green_channel(image1_entry, image2_entry, result_label):
    try: 
        img1 = cv2.imread(image1_entry.get())   
        img2 = cv2.imread(image2_entry.get())

        # Convert images from BGR to RGB
        image1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        image2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # Split images into channels
        _, g1, _ = cv2.split(image1_rgb)
        _, g2, _ = cv2.split(image2_rgb)
        
        hist_r1 = cv2.calcHist([g1], [0], None, [256], [0,256])
        hist_r2 = cv2.calcHist([g2], [0], None, [256], [0,256])
        plt.figure(figsize=(10, 5))
        # Plot red channel and comparison
        plt.plot(hist_r1, color='green', label='Before')
        plt.plot(hist_r2, color='lime', label='After')
        plt.title('Green Channel')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.xlim([0, 256])
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        result_label.configure(text="Select a File!")
    
    
def compare_blue_channel(image1_entry, image2_entry, result_label):
    try: 
        img1 = cv2.imread(image1_entry.get())   
        img2 = cv2.imread(image2_entry.get())

        # Convert images from BGR to RGB
        image1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        image2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # Split images into channels
        b1, _, _ = cv2.split(image1_rgb)
        b2, _, _ = cv2.split(image2_rgb)
        
        hist_r1 = cv2.calcHist([b1], [0], None, [256], [0,256])
        hist_r2 = cv2.calcHist([b2], [0], None, [256], [0,256])
        plt.figure(figsize=(10, 5))
        # Plot red channel and comparison
        plt.plot(hist_r1, color='blue', label='Before')
        plt.plot(hist_r2, color='cyan', label='After')
        plt.title('Green Channel')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.xlim([0, 256])
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        result_label.configure(text="Select a File!")

def psnr_mse(image1_entry, image2_entry, result_label, time): 
    try:
        original = cv2.imread(image1_entry)
        encoded = cv2.imread(image2_entry) 
        mse = np.mean((original - encoded) ** 2)
        if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                    # Therefore PSNR have no importance. 
            psnr = 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse)) 
        
        
        result_label.configure(text="Embedding time = " + str(time) + "\n" +" Peak signal-to-noise ratio = " + str(psnr) + "\n" + " Mean squared error = " + str(mse))
    except Exception as e:
        print(f"Error: {str(e)}")
        result_label.configure(text="Select a File!")

comparison_tab_file_label_before = CTkLabel(comparison_tab, text="Select File Before Hiding Text:")
comparison_tab_file_label_before.pack(pady=5)

comparison_tab_file_entry_before = CTkEntry(comparison_tab,height=20, width=350)
comparison_tab_file_entry_before.pack(pady=5)

comparison_tab_browse_button_before = CTkButton(comparison_tab,text="Browse",cursor="hand2", command=lambda: browseFile_to_compare(comparison_tab_file_entry_before, comparison_tab_result_label))
comparison_tab_browse_button_before.pack(pady=5)




comparison_tab_file_label_after= CTkLabel(comparison_tab, text="Select File After Hiding Text:")
comparison_tab_file_label_after.pack(pady=5)

comparison_tab_file_entry_after = CTkEntry(comparison_tab,height=20, width=350)
comparison_tab_file_entry_after.pack(pady=5)

comparison_tab_browse_button_after = CTkButton(comparison_tab,text="Browse",cursor="hand2", command=lambda: browseFile_to_compare(comparison_tab_file_entry_after, comparison_tab_result_label))
comparison_tab_browse_button_after.pack(pady=(0, 20))



# comparison_tab_button_psnr = CTkButton(comparison_tab,text="PSNR&MSE",cursor="hand2", command=lambda: time_psnr_mse(comparison_tab_file_entry_before, comparison_tab_file_entry_after, comparison_tab_result_label))
# comparison_tab_button_psnr.pack(pady = 5)




comparison_tab_button_red = CTkButton(comparison_tab,text="Red",cursor="hand2", command=lambda: compare_red_channel(comparison_tab_file_entry_before, comparison_tab_file_entry_after, comparison_tab_result_label))
comparison_tab_button_red.pack(pady = 5)

comparison_tab_button_green = CTkButton(comparison_tab,text="Green",cursor="hand2", command=lambda: compare_green_channel(comparison_tab_file_entry_before, comparison_tab_file_entry_after, comparison_tab_result_label))
comparison_tab_button_green.pack(pady = 5)

comparison_tab_button_blue = CTkButton(comparison_tab,text="Blue",cursor="hand2", command=lambda: compare_blue_channel(comparison_tab_file_entry_before, comparison_tab_file_entry_after, comparison_tab_result_label))
comparison_tab_button_blue.pack(pady = 5)

comparison_tab_result_label = CTkLabel(comparison_tab,text="")
comparison_tab_result_label.pack(pady=5)

# if os.path.isfile(hide_text_window_file_entry.get()) & os.path.isfile(reveal_text_window_file_entry.get()):
#     PSNR(hide_text_window_file_entry, reveal_text_window_file_entry, comparison_tab_result_label)

# ---------------------------------------------------------------------------------------
# інструкція для вкладки "Manual"


manual_text = """
### Інструкція з використання додатка StegoProg

#### Приховування тексту у зображенні:

1. Оберіть файл зображення:
   - Клацніть на кнопку "Browse" (Огляд) у вкладці "Hide Text" (Приховати текст).
   - Оберіть зображення, у яке ви хочете приховати текст, та натисніть "Open" (Відкрити).

2. Введіть текст для приховання:
   - У полі для тексту у вкладці "Hide Text" введіть текст, який ви бажаєте приховати у зображенні.

3. Натисніть кнопку "Hide" (Приховати):
   - Після введення тексту натисніть кнопку "Hide".
 Програма приховає текст у зображенні та створить новий файл із збереженим текстом.
#### Розкриття прихованого тексту у зображенні:

1. Оберіть файл зображення:
   - Клацніть на кнопку "Browse" у вкладці "Reveal Text" (Розкрити текст).
   - Оберіть зображення, у якому ви приховали текст, та натисніть "Open".

2. Натисніть кнопку "Reveal Text" (Розкрити текст):
   - Після вибору файлу зображення натисніть кнопку "Reveal Text". 
Програма витягне та відобразить прихований текст із зображення.
#### Попередній перегляд зображень:

- У вкладці "Image Preview" (Попередній перегляд) ви можете порівняти зображення до та після приховання тексту. 
Оберіть файли до та після приховування тексту, і попередній перегляд автоматично оновиться.

#### Порівняння каналів кольору:

- У вкладці "Comparison" (Порівняння) ви можете порівняти різні канали кольору 
(червоний, зелений, синій) зображень до та після приховання тексту.

"""
manual_label = CTkLabel(manual_tab,text= manual_text)
manual_label.pack(padx=10, pady=10)
# ---------------------------------------------------------------------------------------
main_window_exit_button = CTkButton(main_window, text="Exit", font=("Helvetica", 16),command=main_window.destroy, cursor="hand2", corner_radius=15, fg_color="transparent", hover_color="#FF462D", border_color="#FF462D", border_width=2).pack(pady=20, side=BOTTOM)   

main_window.mainloop()
    
    