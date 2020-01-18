import time
import os
import sys
try:
    import cv2
except:
    os.system('pip3 install opencv-python')
    import cv2

try:
    import mss
except:
    os.system('pip3 install mss')
    import mss

import numpy as np
import subprocess


try:
    from PIL import Image
except:
    os.system('pip3 install Pillow')
    from PIL import Image

from datetime import datetime


try:
    import imutils
except:
    os.system('pip3 install imutils')
    import imutils

import threading
import tkinter as tk

root = tk.Tk()
w = root.winfo_screenwidth()
h = root.winfo_screenheight()

print(w, h)
print(os.getcwd())

images = {}
CWD = os.getcwd()
CORNER_WIDTH = 150
CORNER_HEIGHT = 160

# Dimensions of rank train images
RANK_WIDTH = 125
RANK_HEIGHT = 125

# Dimensions of suit train images
SUIT_WIDTH = 125
SUIT_HEIGHT = 125

# IL CODICE FUNZIONA A PATTO CHE NON CI SIANO DUE CARTE UGUALI


def clear():
    for child in root.winfo_children():
        for elem in child.winfo_children():
            elem.destroy()


# @param numbers : sorted list of numbers
# @return restituisce i numeri consecutivi trovati
def check_scala(numbers, lenght=None):

    previous = None
    consecutivi = []
    current_scale = []
    for i, number in enumerate(numbers):
        if i == 0:
            pass
        else:
            if number == previous + 1:
                pass
            elif number == previous:
                continue
            else:
                if len(current_scale) > 1:
                    consecutivi.append(current_scale)
                current_scale = []

        current_scale.append(number)
        previous = number

    if len(current_scale) > 1:
        consecutivi.append(current_scale)

    if lenght is not None:
        max_scale = None
        for scale in consecutivi:
            if max_scale is None:
                max_scale = scale
            else:
                if len(scale) > len(max_scale):
                    max_scale = scale
        if max_scale is None:
            pass
        elif len(max_scale) >= lenght:
            return max_scale[len(max_scale) - lenght:]
        return []

    return consecutivi

# serve per rimuovere il valore 1 dell'asso ( l'asso e considerato come 14 e 1 )
def remove_1(lista):
    for elem in lista:
        if elem == 1:
            lista.remove(elem)


def parse_points(hand_cards, table_cards):

    points = ''
    points_info = ''

    total_cards = hand_cards + table_cards
    hand_num = []
    table_num = []

    cards_dict = {'clubs': [], 'spades': [], 'hearths': [], 'diamonds': []}

    numbers = []

    num_dict = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
                'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'jack': 11, 'queen': 12, 'king': 13, 'ace': 14}

    name_dict = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
                 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'jack', 12: 'queen', 13: 'king', 14: 'ace'}

    for i, card in enumerate(total_cards):
        if card == 'None':
            continue
        card = card.split('-')

        num = 0
        symbol = 1

        # CONVERTE I NUMERI DELLE CARTE IN VALORI INTERI
        if card[num].lower() in num_dict:
            card[num] = num_dict[card[num]]

        if card[symbol] in cards_dict:
            cards_dict[card[symbol]].append(card[num])
            if card[num] == 14:
                cards_dict[card[symbol]].append(1)


        numbers.append(card[num])
        if card[num] == 14:
            numbers.append(1)

    hand_num = numbers[:2]
    table_num = numbers[2:]
    numbers.sort()

    numbers_count = {}
    # CONTA TRA LE CARTE TOTALI IL NUMERO DI CARTE CON LO STESSO VALORE
    for number in numbers:
        if number not in numbers_count:
            numbers_count[number] = 1
        else:
            numbers_count[number] += 1

    coppie = []
    tris = []
    quadruple = []

    for value, count in numbers_count.items():

        if count == 2:
            coppie.append(value)

        elif count == 3:
            tris.append(value)

        elif count == 4:
            quadruple.append(value)

    coppie_count = len(coppie)
    tris_count = len(tris)
    quadruple_count = len(quadruple)

    scale = check_scala(numbers, 5)

    if coppie_count == 1:
        points = 'COPPIA'
        in_hand = False
        if coppie[0] in hand_num:
            in_hand = True

        points_info = points + f' - {coppie[0]} ' + ', IN HAND: ' + str(in_hand)

    if coppie_count > 1:
        copia_coppie = coppie[:]
        c1 = max(copia_coppie)
        copia_coppie.remove(c1)
        c2 = max(copia_coppie)

        in_hand = False
        if c1 in hand_num or c2 in hand_num:
            in_hand = True

        points = 'DOPPIA COPPIA'
        points_info = points + f' - {c2} - {c1}' + ', IN HAND: ' + str(in_hand)

    if tris_count >= 1:
        points = 'TRIS'
        t1 = max(tris)

        in_hand = False
        if t1 in hand_num:
            in_hand = True

        points_info = points + f' - {t1}' + ', IN HAND: ' + str(in_hand)

    if len(scale) == 5:
        points = 'SCALA'
        in_hand = False
        for value in scale:
            if value in hand_num and value not in table_num:
                in_hand = True

        points_info = points + f' - {scale}' + ', IN HAND: ' + str(in_hand)

    if coppie_count >= 1 and tris_count >= 1:
        points = 'FULL'
        c1 = max(coppie)
        t1 = max(tris)

        in_hand = False
        if c1 in hand_num or t1 in hand_num:
            in_hand = True

        points_info = points + f' coppia ({c1}) - tris({t1})' + ', IN HAND: ' + str(in_hand)

    for symbol in cards_dict:
        cards = cards_dict[symbol]
        cards = cards[:]
        remove_1(cards)  # serve per rimuovere il valore 1 dell'asso ( l'asso e considerato come 14 e 1 )
        if len(cards) >= 5:
            in_hand = False
            points = 'COLORE'
            color_symbol = symbol
            for card in cards:
                card_value = name_dict[card]
                card_name = f'{card_value}-{symbol}'
                if card_name in hand_cards:
                    in_hand = True

            points_info = points + f' {symbol} - {cards_dict[symbol]}' + ', IN HAND: ' + str(in_hand)

    if quadruple_count == 1:
        points = 'POKER'
        in_hand = False
        if quadruple[0] in hand_num and quadruple[0] not in table_num:
            in_hand = True
        points_info = points + f' - {quadruple[0]}' + ', IN HAND: ' + str(in_hand)

    if points == 'COLORE':
        color_scale = cards_dict[color_symbol]
        color_scale.sort()
        color_scale = check_scala(color_scale, 5)
        color_scale_ = color_scale[:]
        if len(color_scale) == 5:
            points = 'SCALA COLORE'
            in_hand = False
            remove_1(color_scale)  # serve per rimuovere il valore 1 dell'asso ( l'asso e considerato come 14 e 1 )
            for card in color_scale:
                card_value = name_dict[card]
                card_name = f'{card_value}-{color_symbol}'
                if card_name in hand_cards:
                    in_hand = True
            points_info = points + f' - {color_symbol} {color_scale_}' + ', IN HAND: ' + str(in_hand)
            if color_scale == [10, 11, 12, 13, 14]:
                points = 'SCALA REALE'
                points_info = points + f' - {color_symbol} {color_scale}' + ', IN HAND: ' + str(in_hand)

    return points_info


class My_cards:

    def __init__(self):
        self.hand_cards = []
        self.table_cards = []

        self.update_value = 1

        self.update_hand_counter = 4
        self.update_table_counter = 4

        self.show_points_counter = 0

    def start_loop(self):
        self.root = tk.Tk()

        self.hand_frame = tk.LabelFrame(self.root, text="Hand cards")
        self.table_frame = tk.LabelFrame(self.root, text="Table cards")
        self.points_frame = tk.LabelFrame(self.root, text="Points")

        self.hand_frame.pack()
        self.table_frame.pack()
        self.points_frame.pack()

        self.root.mainloop()

    def points(self):
        for elem in self.points_frame.winfo_children():
            elem.destroy()

        point = parse_points(self.hand_cards, self.table_cards)
        tk.Label(self.points_frame, text=point).pack()

    def send_hand_cards(self, *cards):
        cards = list(cards)
        if self.update_hand_counter >= self.update_value:
            self.update_hand(*cards)
            self.update_hand_counter = 0

        if cards != self.hand_cards:
            self.update_hand_counter += 1
        else:
            self.update_hand_counter = 0

    def send_table_cards(self, *cards):
        cards = list(cards)

        if self.update_table_counter >= self.update_value:
            self.update_table(*cards)
            self.update_table_counter = 0

        if cards != self.table_cards:
            self.update_table_counter += 1
        else:
            self.update_table_counter = 0

    def update_hand(self, *cards):
        cards = list(cards)
        self.hand_cards = cards

        for elem in self.hand_frame.winfo_children():
            elem.destroy()

        if len(self.hand_cards) == 0:
            tk.Label(self.hand_frame, text='None').pack()
        else:
            for index, card in enumerate(self.hand_cards):
                tk.Label(self.hand_frame, text=self.hand_cards[index]).pack()

    def update_table(self, *cards):
        cards = list(cards)
        self.table_cards = cards

        for elem in self.table_frame.winfo_children():
            elem.destroy()

        if len(self.table_cards) == 0:
            tk.Label(self.table_frame, text='None').pack()
        else:
            for index, card in enumerate(self.table_cards):
                tk.Label(self.table_frame, text=self.table_cards[index]).pack()

        self.points()




current_game = My_cards()




def save_images(num_sized, symbol_sized):
    if input('Want save num? ') in ['Y', 'y']:
        file_name = input('FIle name: ')
        save_path = 'cards/numbers/' + file_name + '.png'
        while os.path.exists(save_path):
            file_name = input('FIle name: ')
            save_path = 'cards/numbers/' + file_name + '.png'

        with open(save_path, 'w') as file:
            pass

        cv2.imwrite(save_path, num_sized)

    if input('Want save symbol? ') in ['Y', 'y']:
        file_name = input('FIle name: ')
        save_path = 'cards/symbols/' + file_name + '.png'
        while os.path.exists(save_path):
            file_name = input('FIle name: ')
            save_path = 'cards/symbols/' + file_name + '.png'

        with open(save_path, 'w') as file:
            pass
        cv2.imwrite(save_path, symbol_sized)


def populate_images():
    global images
    for folder in os.listdir(f'{CWD}/cards'):
        images[folder] = {}
        print(folder)
        for file_name in os.listdir(f'{CWD}/cards/{folder}'):
            file = f'{CWD}/cards/{folder}/{file_name}'
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if folder == 'numbers':
                image = cv2.resize(image, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
            elif folder == 'symbols':
                image = cv2.resize(image, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
            name = file_name.split('_')[0].lower()
            if not name in images[folder]:
                images[folder][name] = []

            images[folder][name].append(image)




def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def preprocess(img):
    img = cv2.resize(img, (450, 450))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
    return thresh


def is_in_array(array, list):
    for elem in list:
        if np.array_equal(array, elem):
            return True
    return False


def bincount_app(a):
    a2D = a.reshape(-1, a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)

def find_contours(image, max_peri, min_peri):

    contours_dict = dict()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 1000)
    flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    contourns_to_filter = []

    # filter the cards contour
    for i in range(len(contours)):

        card = contours[i]
        peri = cv2.arcLength(card, True)
        if (max_peri > peri > min_peri) and (not is_in_array(card, contourns_to_filter)):
            contourns_to_filter.append(card)

    return contourns_to_filter

def elaborate_contour(image, contour, contours_list):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
    rect = cv2.boundingRect(contour)
    x, y, w, h = rect
    image_to_compute = np.array([[0, 0], [449, 0], [449, 449], [0, 449]], np.float32)
    if len(rect) != 4:
        return
    else:
        temp_rect = np.zeros((4, 2), dtype="float32")
        s = np.sum(approx, axis=2)
        tl = approx[np.argmin(s)]
        br = approx[np.argmax(s)]
        diff = np.diff(approx, axis=-1)
        tr = approx[np.argmin(diff)]
        bl = approx[np.argmax(diff)]

        # Need to create an array listing points in order of
        # [top left, top right, bottom right, bottom left]
        # before doing the perspective transform
        if w <= 0.8 * h:  # If card is vertically oriented
            temp_rect[0] = tl
            temp_rect[1] = tr
            temp_rect[2] = br
            temp_rect[3] = bl

        if w >= 1.2 * h:  # If card is horizontally oriented
            temp_rect[0] = bl
            temp_rect[1] = tl
            temp_rect[2] = tr
            temp_rect[3] = br

        # If the card is 'diamond' oriented, a different algorithm
        # has to be used to identify which point is top left, top right
        # bottom left, and bottom right.

        if w > 0.8 * h and w < 1.2 * h and (1.2 * h - 0.8 * h):  # If card is diamond oriented
            # If furthest left point is higher than furthest right point,
            # card is tilted to the left.q
            if approx[1][0][1] <= approx[3][0][1]:
                # If card is titled to the left, approxPolyDP returns points
                # in this order: top right, top left, bottom left, bottom right
                temp_rect[0] = approx[1][0]  # Top left
                temp_rect[1] = approx[0][0]  # Top right
                temp_rect[2] = approx[3][0]  # Bottom right
                temp_rect[3] = approx[2][0]  # Bottom left

            # If furthest left point is lower than furthest right point,
            # card is tilted to the right
            if approx[1][0][1] > approx[3][0][1]:
                # If card is titled to the right, approxPolyDP returns points
                # in this order: top left, bottom left, bottom right, top right
                temp_rect[0] = approx[0][0]  # Top left
                temp_rect[1] = approx[3][0]  # Top right
                temp_rect[2] = approx[2][0]  # Bottom right
                temp_rect[3] = approx[1][0]  # Bottom left

        transform = cv2.getPerspectiveTransform(temp_rect, image_to_compute)

    warp = cv2.warpPerspective(image, transform, (450, 450))
    corner = warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    corner_zoom = cv2.resize(corner, (0, 0), fx=4, fy=4)
    corner_zoom_gray = cv2.cvtColor(corner_zoom, cv2.COLOR_BGR2GRAY)
    corner_zoom_blur = cv2.GaussianBlur(corner_zoom_gray, (5, 5), 2)
    flag, corner_zoom_thresh = cv2.threshold(corner_zoom_blur, 150, 255, cv2.THRESH_BINARY)
    retval, corner_zoom_thresh = cv2.threshold(corner_zoom_thresh, 200, 255, cv2.THRESH_BINARY_INV)

    num = corner_zoom_thresh[0:340, 0:]
    symbol = corner_zoom_thresh[350:, 0:]

    num_sized = ''
    symbol_sized = ''

    # Find rank contour and bounding rectangle, isolate and find largest contour
    num_contourn, hier = cv2.findContours(num, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num_contourn = sorted(num_contourn, key=cv2.contourArea, reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query rank
    # image to match dimensions of the train rank image
    if len(num_contourn) != 0:
        rect1 = cv2.boundingRect(num_contourn[0])
        x1, y1, w1, h1 = rect1
        num_roi = num[y1: y1 + h1, x1:x1 + w1]
        num_sized = cv2.resize(num_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
        #cv2.imshow('num', num_sized)

    # Find suit contour and bounding rectangle, isolate and find largest contour
    symbol_contourn, hier = cv2.findContours(symbol, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    symbol_contourn = sorted(symbol_contourn, key=cv2.contourArea, reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query suit
    # image to match dimensions of the train suit image
    if len(symbol_contourn) != 0:
        rect2 = cv2.boundingRect(symbol_contourn[0])
        x2, y2, w2, h2 = rect2
        symbol_roi = symbol[y2:y2 + h2, x2:x2 + w2]
        symbol_sized = cv2.resize(symbol_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        #cv2.imshow('symbol', symbol_sized)

    #cv2.waitKey(0)

    symbol_find = 'unknown'
    num_find = 'unknown'

    symbol_diff = None
    num_diff = None

    if num_sized != '':
        for num_name in images['numbers']:
            images_num = images['numbers'][num_name]
            for image_num in images_num:
                diff = cv2.absdiff(num_sized, image_num)
                diff = cv2.GaussianBlur(diff, (5, 5), 5)
                flag, diff = cv2.threshold(diff, 150, 255, cv2.THRESH_BINARY)
                diff = np.sum(diff)
                if num_diff is None:
                    num_diff = diff
                    num_find = num_name
                elif diff < num_diff:
                    num_diff = diff
                    num_find = num_name

    if symbol_sized != '':
        for symbol_name in images['symbols']:
            images_symbol = images['symbols'][symbol_name]
            for image_symbol in images_symbol:
                diff = cv2.absdiff(symbol_sized, image_symbol)
                diff = cv2.GaussianBlur(diff, (5, 5), 5)
                flag, diff = cv2.threshold(diff, 150, 255, cv2.THRESH_BINARY)

                diff = np.sum(diff)
                if symbol_diff is None:
                    symbol_diff = diff
                    symbol_find = symbol_name
                elif diff < symbol_diff:
                    symbol_diff = diff
                    symbol_find = symbol_name

    contours_list.append([contour, symbol_diff, num_diff, symbol_find, num_find])


def find_my_cards(image):

    my_cards = []

    p1 = [1000, 614]
    p2 = [1037, 638]
    p3 = [1018, 683]
    p4 = [983, 673]
    card1 = np.array([[p3], [p4], [p1], [p2]], np.float32)

    p1 = [864, 660]
    p2 = [892, 652]
    p3 = [917, 709]
    p4 = [884, 722]
    card2 = np.array([[p4], [p3], [p2], [p1]], np.float32)

    image_to_compute = np.array([[0, 0], [449, 0], [449, 449], [0, 449]], np.float32)
    cards = [card1, card2]

    for card in cards:
        approx = card
        temp_rect = np.zeros((4, 2), dtype="float32")
        s = np.sum(approx, axis=2)
        tl = approx[np.argmin(s)]
        br = approx[np.argmax(s)]
        diff = np.diff(approx, axis=-1)
        tr = approx[np.argmin(diff)]
        bl = approx[np.argmax(diff)]


        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl


        transform = cv2.getPerspectiveTransform(temp_rect, image_to_compute)

        warp = cv2.warpPerspective(image, transform, (450, 450))
        warp = rotateImage(warp, 5)
        corner = warp

        corner_zoom_gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        corner_zoom_blur = cv2.GaussianBlur(corner_zoom_gray, (5, 5), 2)
        flag, corner_zoom_thresh = cv2.threshold(corner_zoom_blur, 150, 255, cv2.THRESH_BINARY)
        retval, corner_zoom_thresh = cv2.threshold(corner_zoom_thresh, 200, 255, cv2.THRESH_BINARY_INV)
        #cv2.imshow('warp', warp)
        num = corner_zoom_thresh[0:260, 0:]
        symbol = corner_zoom_thresh[210:, 0:]


        num_sized = ''
        symbol_sized = ''

        # Find rank contour and bounding rectangle, isolate and find largest contour
        num_contourn, hier = cv2.findContours(num, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        num_contourn = sorted(num_contourn, key=cv2.contourArea, reverse=True)

        # Find bounding rectangle for largest contour, use it to resize query rank
        # image to match dimensions of the train rank image
        if len(num_contourn) != 0:
            rect1 = cv2.boundingRect(num_contourn[0])
            x1, y1, w1, h1 = rect1
            num_roi = num[y1: y1 + h1, x1:x1 + w1]
            num_sized = cv2.resize(num_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
            num_sized = rotateImage(num_sized, -6)
            #cv2.imshow('num', num_sized)

        # Find suit contour and bounding rectangle, isolate and find largest contour
        symbol_contourn, hier = cv2.findContours(symbol, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        symbol_contourn = sorted(symbol_contourn, key=cv2.contourArea, reverse=True)

        # Find bounding rectangle for largest contour, use it to resize query suit
        # image to match dimensions of the train suit image
        if len(symbol_contourn) != 0:
            rect2 = cv2.boundingRect(symbol_contourn[0])
            x2, y2, w2, h2 = rect2
            symbol_roi = symbol[y2:y2 + h2, x2:x2 + w2]
            symbol_sized = cv2.resize(symbol_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
            #v2.imshow('symbol', symbol_sized)

        #cv2.waitKey(0)

        symbol_find = 'unknown'
        num_find = 'unknown'

        symbol_diff = None
        num_diff = None

        if num_sized != '':
            for num_name in images['numbers']:
                images_num = images['numbers'][num_name]
                for image_num in images_num:

                    diff = cv2.absdiff(num_sized, image_num)
                    diff = cv2.GaussianBlur(diff, (5, 5), 5)
                    flag, diff = cv2.threshold(diff, 150, 255, cv2.THRESH_BINARY)
                    diff = np.sum(diff)
                    if num_diff is None:
                        num_diff = diff
                        num_find = num_name
                    elif diff < num_diff:
                        num_diff = diff
                        num_find = num_name

        if symbol_sized != '':
            for symbol_name in images['symbols']:
                images_symbol = images['symbols'][symbol_name]
                for image_symbol in images_symbol:
                    diff = cv2.absdiff(symbol_sized, image_symbol)
                    diff = cv2.GaussianBlur(diff, (5, 5), 5)
                    flag, diff = cv2.threshold(diff, 150, 255, cv2.THRESH_BINARY)

                    diff = np.sum(diff)
                    if symbol_diff is None:
                        symbol_diff = diff
                        symbol_find = symbol_name
                    elif diff < symbol_diff:
                        symbol_diff = diff
                        symbol_find = symbol_name
        min_comparison = 750_000
        if symbol_diff is not None and num_diff is not None:

            if symbol_diff < min_comparison and num_diff < min_comparison:

                my_cards.append(f'{num_find}-{symbol_find}')
            else:
                my_cards.append('None')

    font = cv2.FONT_HERSHEY_SIMPLEX

    current_game.send_hand_cards(my_cards[0], my_cards[1])

    cv2.putText(image, 'card 1: ' + my_cards[0], (10, 200), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'card 2: ' + my_cards[1], (10, 250), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


    return image

def find_cards(image):

    contours = find_contours(image, 1000, 500)
    contours_list = []
    contours_to_draw = []



    t = None
    print(len(contours))
    for i, contour in enumerate(contours):
        #t = threading.Thread(target=elaborate_contour, args=[image, contour, contours_list], name=str(i))
        #t.start()

        elaborate_contour(image, contour, contours_list)

    if t is not None:
        t.join()

    font = cv2.FONT_HERSHEY_SIMPLEX
    found_cards = []
    for item in contours_list:

        min_comparison = 700_000
        contour = item[0]
        symbol_diff = item[1]
        num_diff = item[2]
        symbol_find = item[3]
        num_find = item[4]
        if symbol_diff is not None and num_diff is not None:
            if symbol_diff < min_comparison and num_diff < min_comparison:
                contours_to_draw.append(contour)
                M = cv2.moments(contour)

                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)

                found_cards.append(num_find + '-' + symbol_find)

                cv2.putText(image, (num_find + ' of'), (cX - 30, cY - 10), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(image, (num_find + ' of'), (cX - 30, cY - 10), font, 0.5, (50, 200, 200), 2, cv2.LINE_AA)

                cv2.putText(image, symbol_find, (cX - 30, cY + 25), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(image, symbol_find, (cX - 30, cY + 25), font, 0.5, (50, 200, 200), 2, cv2.LINE_AA)

    current_game.send_table_cards(*found_cards)
    image = cv2.drawContours(image, contours_to_draw, -1, (0, 0, 255), 5)

    return image

def draw_coords(event,x,y,flags,param):
    print('here')
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)

populate_images()
loop = threading.Thread(target=current_game.start_loop)
loop.start()


with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 0, "left": 0, "width": w, "height": h}
    print(monitor)

    while True:
        last_time = time.time()

        screen = sct.grab(monitor)

        img = Image.frombytes("RGB", screen.size, screen.bgra, "raw", "BGRX")


        img = np.array(img)
        img = find_my_cards(img)
        img = find_cards(img)
        # Display the picture
        font = cv2.FONT_HERSHEY_COMPLEX
        # date = str(datetime.now())
        date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(img, date, (w//2 - 100, 120), font, 0.5, (200, 255, 50), 1, cv2.LINE_AA)

        # cv2.circle(img, (850, 615), 5, (255, 0, 0), -1)
        current_game.points()
        time.sleep(2)




        #print("fps: {}".format(1 / (time.time() - last_time)))

        # Press "q" to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

cv2.destroyAllWindows()
