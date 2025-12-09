import os
import sys
import io
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, Menu
import threading
import mss
import numpy as np
from PIL import Image, ImageTk
import pytesseract
import cv2
import configparser
from openai import OpenAI

# ==================== 强制 UTF-8 ====================
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ==================== 读取配置文件 ====================
config = configparser.ConfigParser()
try:
    config.read('config.ini', encoding='utf-8')
except Exception as e:
    print(f"❌ 无法读取 config.ini: {e}")
    input("按回车退出...")
    sys.exit(1)

API_ADDRESS = config.get('Settings', 'api_address', fallback='https://api.deepseek.com')
API_KEY = config.get('Settings', 'api_key', fallback='').strip()
MODEL_NAME = config.get('Settings', 'model_name', fallback='deepseek-chat')
PRE_PROMPT = config.get('Settings', 'pre_prompt', fallback='将下面的文本翻译成简体中文：')
SYSTEM_PROMPT = config.get('Settings', 'system_prompt', fallback='你是一个翻译助手。')
TEMPERATURE = float(config.get('Settings', 'temperature', fallback='1.0'))
CONTEXT_NUM = int(config.get('Settings', 'context_num', fallback='5'))

# ==================== 主题颜色定义 ====================
THEMES = {
    "白色": {
        "bg": "#FFFFFF",
        "fg": "#000000",
        "button_bg": "#F0F0F0",
        "button_fg": "#000000",
        "input_bg": "#FFFFFF",
        "input_fg": "#000000",
        "label_bg": "#F8F8F8",
        "text_bg": "#FFFFFF",
        "text_fg": "#000000",
        "border": "#CCCCCC"
    },
    "黑色": {
        "bg": "#000000",
        "fg": "#FFFFFF",
        "button_bg": "#333333",
        "button_fg": "#FFFFFF",
        "input_bg": "#1A1A1A",
        "input_fg": "#FFFFFF",
        "label_bg": "#222222",
        "text_bg": "#1A1A1A",
        "text_fg": "#FFFFFF",
        "border": "#444444"
    },
    "卡其色": {
        "bg": "#BDB76B",
        "fg": "#2F4F4F",
        "button_bg": "#D3D3A6",
        "button_fg": "#2F4F4F",
        "input_bg": "#FFFFF0",
        "input_fg": "#2F4F4F",
        "label_bg": "#DCDCDC",
        "text_bg": "#FFFFF0",
        "text_fg": "#2F4F4F",
        "border": "#8B7355"
    },
    "暗灰色": {
        "bg": "#696969",
        "fg": "#FFFFFF",
        "button_bg": "#808080",
        "button_fg": "#FFFFFF",
        "input_bg": "#4F4F4F",
        "input_fg": "#FFFFFF",
        "label_bg": "#5A5A5A",
        "text_bg": "#4F4F4F",
        "text_fg": "#FFFFFF",
        "border": "#404040"
    }
}

# ==================== 设置 Tesseract OCR 路径 ====================
# 设置 Tesseract OCR 可执行文件路径
TESSERACT_EXE_PATH = r"E:\Tesseract OCR\tesseract.exe"
# 设置 Tesseract OCR 数据目录路径
TESSDATA_PREFIX = r"E:\Tesseract OCR\tessdata"

# 检查并设置 Tesseract OCR 路径
if os.path.exists(TESSERACT_EXE_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE_PATH
    print(f"✅ 已设置 Tesseract OCR 可执行文件路径: {TESSERACT_EXE_PATH}")
else:
    print(f"⚠️  警告: Tesseract OCR 可执行文件路径不存在: {TESSERACT_EXE_PATH}")

# 检查并设置 TESSDATA_PREFIX 环境变量
if os.path.exists(TESSDATA_PREFIX):
    # 设置环境变量，让 Tesseract 知道语言数据的位置
    os.environ['TESSDATA_PREFIX'] = TESSDATA_PREFIX
    print(f"✅ 已设置 TESSDATA_PREFIX 环境变量: {TESSDATA_PREFIX}")

    # 检查语言数据文件是否存在
    eng_traineddata = os.path.join(TESSDATA_PREFIX, "eng.traineddata")
    if os.path.exists(eng_traineddata):
        print(f"✅ 找到英文语言数据文件: {eng_traineddata}")
    else:
        print(f"⚠️  警告: 未找到英文语言数据文件 eng.traineddata")
else:
    print(f"⚠️  警告: TESSDATA_PREFIX 路径不存在: {TESSDATA_PREFIX}")

# ✅ 关键检查：API Key 是否为空
if not API_KEY:
    print("❌ 错误：config.ini 中的 api_key 为空！")
    print("请前往 https://platform.deepseek.com/ 获取 API Key 并填入 config.ini")
    input("按回车退出...")
    sys.exit(1)


class GameTranslationAssistant:
    def __init__(self):
        # 检查 Tesseract OCR 是否可用
        self.ocr_available = self.check_tesseract_available()

        self.root = tk.Tk()
        self.root.title("🎮 游戏翻译助手")

        # 当前主题
        self.current_theme = "白色"  # 默认主题

        # 窗口设置
        self.root.wm_attributes("-alpha", 0.95)
        self.root.wm_attributes("-topmost", True)

        # 去掉默认标题栏，使用自定义标题栏
        self.root.overrideredirect(True)

        # 初始窗口大小和位置
        self.window_width = 400
        self.window_height = 600
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.x_position = self.screen_width - self.window_width - 100
        self.y_position = 100

        self.root.geometry(f"{self.window_width}x{self.window_height}+{self.x_position}+{self.y_position}")

        # 初始化客户端
        try:
            self.client = OpenAI(
                api_key=API_KEY,
                base_url=API_ADDRESS,
                timeout=30
            )
        except Exception as e:
            messagebox.showerror("初始化失败", f"DeepSeek 客户端初始化失败:\n{e}")
            self.status_var = tk.StringVar(value="⚠️ API 初始化失败")
            self.setup_ui()
            return

        self.is_selecting = False
        self.selection_start = None
        self.selection_rect = None
        self.screenshot_img = None  # 保存原始截图

        # 窗口调整大小相关变量
        self.resizing = False
        self.resize_start_x = 0
        self.resize_start_y = 0
        self.resize_start_width = 0
        self.resize_start_height = 0

        self.setup_ui()
        self.setup_hotkeys()

        # 应用初始主题
        self.apply_theme(self.current_theme)

    def check_tesseract_available(self):
        """检查 Tesseract OCR 是否可用"""
        try:
            # 尝试获取 Tesseract 版本
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract OCR 版本: {version}")
            return True
        except Exception as e:
            print(f"❌ Tesseract OCR 不可用: {e}")
            print("\n请检查以下配置：")
            print(f"1. Tesseract 可执行文件路径: {TESSERACT_EXE_PATH}")
            print(f"2. TESSDATA_PREFIX 环境变量: {TESSDATA_PREFIX}")
            print(f"3. 确保 {TESSDATA_PREFIX} 目录中包含 eng.traineddata 文件")
            return False

    def setup_ui(self):
        # 创建主容器
        self.main_container = tk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # 自定义标题栏
        self.title_bar = tk.Frame(self.main_container, height=30)
        self.title_bar.pack(fill=tk.X)
        self.title_bar.pack_propagate(False)

        # 标题栏内容
        title_label = tk.Label(self.title_bar, text="🎮 游戏翻译助手", font=("Microsoft YaHei", 10, "bold"))
        title_label.pack(side=tk.LEFT, padx=10)

        # 主题选择菜单按钮
        theme_button = tk.Button(self.title_bar, text="🎨 主题", command=self.show_theme_menu, bd=0)
        theme_button.pack(side=tk.RIGHT, padx=5)

        # 关闭按钮
        close_button = tk.Button(self.title_bar, text="×", width=2, command=self.root.quit, bd=0, font=("Arial", 12))
        close_button.pack(side=tk.RIGHT, padx=5)

        # 最小化按钮
        minimize_button = tk.Button(self.title_bar, text="_", width=2, command=self.minimize_window, bd=0)
        minimize_button.pack(side=tk.RIGHT, padx=5)

        # 绑定标题栏拖动事件
        self.title_bar.bind("<ButtonPress-1>", self.start_move)
        self.title_bar.bind("<B1-Motion>", self.do_move)
        title_label.bind("<ButtonPress-1>", self.start_move)
        title_label.bind("<B1-Motion>", self.do_move)

        # 主内容区域
        main_frame = tk.Frame(self.main_container)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 如果 OCR 不可用，显示警告
        if not self.ocr_available:
            warning_frame = tk.Frame(main_frame)
            warning_frame.pack(fill=tk.X, pady=(0, 5))
            tk.Label(warning_frame,
                     text="⚠️ OCR未正确配置 - 截图功能可能无法识别文字",
                     font=("Microsoft YaHei", 9, "bold")).pack(fill=tk.X)

        tk.Label(main_frame, text="📸 F10: 截图翻译 | ✏️ 手动输入").pack(anchor=tk.W, pady=(0, 5))

        # 输入区域
        input_frame = tk.LabelFrame(main_frame, text="输入", padx=5, pady=5)
        input_frame.pack(fill=tk.BOTH, expand=True)

        self.input_text = scrolledtext.ScrolledText(input_frame, height=7, wrap=tk.WORD, font=("Microsoft YaHei", 9))
        self.input_text.pack(fill=tk.BOTH, expand=True)

        btn_frame = tk.Frame(input_frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))
        self.translate_button = tk.Button(btn_frame, text="🔄 翻译", command=self.translate_text)
        self.translate_button.pack(side=tk.LEFT, padx=(0, 5))
        self.clear_button = tk.Button(btn_frame, text="🗑️ 清空", command=self.clear_all)
        self.clear_button.pack(side=tk.LEFT)

        # 预览区域
        preview_frame = tk.LabelFrame(main_frame, text="截图预览", padx=5, pady=5)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.preview_label = tk.Label(preview_frame, text="无截图", anchor="center")
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # 输出区域
        output_frame = tk.LabelFrame(main_frame, text="翻译结果", padx=5, pady=5)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.output_text = scrolledtext.ScrolledText(output_frame, height=10, wrap=tk.WORD, font=("Microsoft YaHei", 9))
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪 - 按F10截图或手动输入")
        if not self.ocr_available:
            self.status_var.set("就绪 - OCR未正确配置，只能手动输入")

        self.status_label = tk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padx=5)
        self.status_label.pack(fill=tk.X, pady=(5, 0))

        # 窗口调整大小手柄（右下角的小三角形）
        self.resize_handle = tk.Canvas(self.main_container, width=15, height=15, bg='gray', highlightthickness=0)
        self.resize_handle.place(relx=1.0, rely=1.0, anchor='se')
        self.resize_handle.create_polygon(15, 0, 0, 15, 15, 15, fill='darkgray')

        # 绑定调整大小事件
        self.resize_handle.bind("<ButtonPress-1>", self.start_resize)
        self.resize_handle.bind("<B1-Motion>", self.do_resize)
        self.resize_handle.bind("<ButtonRelease-1>", self.stop_resize)

        # 允许窗口边缘调整大小
        self.root.bind("<Configure>", self.on_window_resize)

    def show_theme_menu(self):
        """显示主题选择菜单"""
        menu = Menu(self.root, tearoff=0)
        for theme_name in THEMES.keys():
            menu.add_command(label=theme_name, command=lambda t=theme_name: self.apply_theme(t))
        menu.post(self.title_bar.winfo_rootx() + 100, self.title_bar.winfo_rooty() + 30)

    def apply_theme(self, theme_name):
        """应用主题"""
        if theme_name not in THEMES:
            return

        self.current_theme = theme_name
        theme = THEMES[theme_name]

        # 应用主题到所有控件
        self.root.configure(bg=theme["bg"])
        self.main_container.configure(bg=theme["bg"])
        self.title_bar.configure(bg=theme["bg"])

        # 标题栏子控件
        for widget in self.title_bar.winfo_children():
            if isinstance(widget, tk.Label):
                widget.configure(bg=theme["bg"], fg=theme["fg"])
            elif isinstance(widget, tk.Button):
                widget.configure(bg=theme["button_bg"], fg=theme["button_fg"])

        # 主内容区域
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame):
                widget.configure(bg=theme["bg"])

        # 状态栏
        self.status_label.configure(bg=theme["label_bg"], fg=theme["fg"])

        # 文本区域
        self.input_text.configure(bg=theme["input_bg"], fg=theme["input_fg"],
                                  insertbackground=theme["input_fg"])
        self.output_text.configure(bg=theme["text_bg"], fg=theme["text_fg"],
                                   insertbackground=theme["text_fg"])

        # 按钮
        self.translate_button.configure(bg=theme["button_bg"], fg=theme["button_fg"])
        self.clear_button.configure(bg=theme["button_bg"], fg=theme["button_fg"])

        # 调整大小手柄
        self.resize_handle.configure(bg=theme["bg"])
        self.resize_handle.delete("all")
        self.resize_handle.create_polygon(15, 0, 0, 15, 15, 15, fill=theme["border"])

    def minimize_window(self):
        """最小化窗口"""
        self.root.iconify()

    def start_move(self, event):
        """开始移动窗口"""
        self._x = event.x
        self._y = event.y

    def do_move(self, event):
        """移动窗口"""
        deltax = event.x - self._x
        deltay = event.y - self._y
        x = self.root.winfo_x() + deltax
        y = self.root.winfo_y() + deltay
        self.root.geometry(f"+{x}+{y}")

    def start_resize(self, event):
        """开始调整窗口大小"""
        self.resizing = True
        self.resize_start_x = event.x_root
        self.resize_start_y = event.y_root
        self.resize_start_width = self.root.winfo_width()
        self.resize_start_height = self.root.winfo_height()

    def do_resize(self, event):
        """调整窗口大小"""
        if not self.resizing:
            return

        # 计算新的宽度和高度
        delta_x = event.x_root - self.resize_start_x
        delta_y = event.y_root - self.resize_start_y

        new_width = max(300, self.resize_start_width + delta_x)  # 最小宽度300
        new_height = max(400, self.resize_start_height + delta_y)  # 最小高度400

        # 应用新的窗口大小
        self.root.geometry(f"{new_width}x{new_height}")

    def stop_resize(self, event):
        """停止调整窗口大小"""
        self.resizing = False

    def on_window_resize(self, event):
        """窗口大小改变时更新UI"""
        if event.widget == self.root:
            self.window_width = event.width
            self.window_height = event.height

    def setup_hotkeys(self):
        self.root.bind("<F10>", lambda e: self.start_screenshot())
        self.root.bind("<Escape>", lambda e: self.root.quit())

    def start_screenshot(self):
        if not self.ocr_available:
            messagebox.showwarning(
                "OCR 未配置",
                "Tesseract OCR 未正确配置，截图功能可能无法识别文字。\n\n"
                "请检查以下配置：\n"
                f"1. Tesseract 可执行文件: {TESSERACT_EXE_PATH}\n"
                f"2. TESSDATA_PREFIX 环境变量: {TESSDATA_PREFIX}\n"
                "3. 确保 tessdata 目录中包含 eng.traineddata 文件"
            )
            # 用户可以选择继续（只截图不识别）
            response = messagebox.askyesno("继续截图", "OCR未配置，继续截图但无法识别文字？")
            if not response:
                return

        self.status_var.set("准备截图...点击并拖拽选择区域")
        self.root.withdraw()

        with mss.mss() as sct:
            # 获取所有显示器的边界框
            all_monitors = sct.monitors

            # 计算所有显示器的组合边界
            left = min(monitor['left'] for monitor in all_monitors)
            top = min(monitor['top'] for monitor in all_monitors)
            right = max(monitor['left'] + monitor['width'] for monitor in all_monitors)
            bottom = max(monitor['top'] + monitor['height'] for monitor in all_monitors)
            width = right - left
            height = bottom - top

            # 创建覆盖所有显示器的截图窗口
            self.screenshot_window = tk.Toplevel()
            self.screenshot_window.attributes("-alpha", 0.3)
            self.screenshot_window.attributes("-topmost", True)
            self.screenshot_window.overrideredirect(True)
            self.screenshot_window.config(bg='black')

            # 设置窗口位置和大小以覆盖所有显示器
            self.screenshot_window.geometry(f"{width}x{height}+{left}+{top}")

            # 截取整个虚拟屏幕
            screenshot = sct.grab(all_monitors[0])  # monitor 0 是整个虚拟屏幕
            self.screenshot_img = np.array(screenshot)  # 保存原始截图数据
            img_rgb = cv2.cvtColor(self.screenshot_img, cv2.COLOR_BGRA2RGB)
            self.fullscreen_pil = Image.fromarray(img_rgb)

            self.canvas = tk.Canvas(self.screenshot_window, cursor="cross", bg="black", highlightthickness=0)
            self.canvas.pack(fill=tk.BOTH, expand=True)

            # 创建一个全尺寸的矩形来模拟透明覆盖
            self.canvas.create_rectangle(0, 0, width, height, fill="black", stipple="gray50")

            self.canvas.bind("<ButtonPress-1>", self.on_select_start)
            self.canvas.bind("<B1-Motion>", self.on_select_motion)
            self.canvas.bind("<ButtonRelease-1>", self.on_select_end)
            self.canvas.bind("<Escape>", lambda e: self.cancel_screenshot())

    def on_select_start(self, event):
        self.is_selecting = True
        self.selection_start = (event.x, event.y)

    def on_select_motion(self, event):
        if self.is_selecting and self.selection_start:
            if self.selection_rect:
                self.canvas.delete(self.selection_rect)
            x0, y0 = self.selection_start
            x1, y1 = event.x, event.y
            self.selection_rect = self.canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=2, dash=(4, 2))

            # 显示选择区域的尺寸
            self.canvas.delete("size_label")
            width = abs(x1 - x0)
            height = abs(y1 - y0)
            label_x = min(x0, x1) + width / 2
            label_y = min(y0, y1) - 20
            if label_y < 0:
                label_y = min(y0, y1) + height + 20
            self.canvas.create_text(label_x, label_y,
                                    text=f"{width} x {height}",
                                    fill="yellow", font=("Arial", 10, "bold"),
                                    tags="size_label")

    def on_select_end(self, event):
        if not (self.is_selecting and self.selection_start):
            return

        x0, y0 = self.selection_start
        x1, y1 = event.x, event.y
        left, top = min(x0, x1), min(y0, y1)
        right, bottom = max(x0, x1), max(y0, y1)

        if (right - left) < 10 or (bottom - top) < 10:
            self.cancel_screenshot()
            self.status_var.set("区域太小，请重新选择")
            return

        # 获取虚拟屏幕的边界信息
        with mss.mss() as sct:
            all_monitors = sct.monitors
            virtual_left = min(monitor['left'] for monitor in all_monitors)
            virtual_top = min(monitor['top'] for monitor in all_monitors)

        # 计算实际屏幕坐标（考虑多个显示器）
        actual_left = virtual_left + left
        actual_top = virtual_top + top
        actual_right = virtual_left + right
        actual_bottom = virtual_top + bottom

        # 使用mss直接截取选定的区域 - 使用原始分辨率
        with mss.mss() as sct:
            monitor = {
                "left": int(actual_left),
                "top": int(actual_top),
                "width": int(actual_right - actual_left),
                "height": int(actual_bottom - actual_top)
            }
            screenshot = sct.grab(monitor)
            img_array = np.array(screenshot)
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
            cropped_img = Image.fromarray(img_rgb)

        # 保存原始截图用于OCR识别
        self.original_screenshot = cropped_img.copy()

        # 显示预览
        self.show_preview(cropped_img)

        # 如果OCR可用，则进行文字识别
        if self.ocr_available:
            self.recognize_text(cropped_img)
        else:
            # OCR不可用，只显示预览
            self.input_text.delete("1.0", tk.END)
            self.input_text.insert(tk.END, "⚠️ OCR未正确配置，请手动输入要翻译的文字")
            self.status_var.set("截图完成，但无法识别文字（OCR未配置）")

        self.screenshot_window.destroy()
        self.restore_main_window()
        self.is_selecting = False

    def cancel_screenshot(self):
        if hasattr(self, 'screenshot_window'):
            self.screenshot_window.destroy()
        self.restore_main_window()
        self.is_selecting = False
        self.status_var.set("截图已取消")

    def restore_main_window(self):
        self.root.deiconify()
        self.root.lift()
        self.root.wm_attributes("-topmost", True)

    def show_preview(self, image):
        max_w, max_h = 300, 150
        w, h = image.size
        ratio = min(max_w / w, max_h / h)
        new_size = (int(w * ratio), int(h * ratio))
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        self.preview_tk = ImageTk.PhotoImage(resized)
        self.preview_label.config(image=self.preview_tk, text="")
        self.preview_label.image = self.preview_tk

    def preprocess_image(self, image):
        """改进的图像预处理"""
        img = np.array(image)

        # 转换为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # 方法1: 自适应阈值
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # 方法2: 大津二值化
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 去噪
        denoised_adaptive = cv2.medianBlur(adaptive_thresh, 3)
        denoised_otsu = cv2.medianBlur(otsu_thresh, 3)

        # 返回处理后的图像
        return {
            "adaptive": denoised_adaptive,
            "otsu": denoised_otsu,
            "original_gray": gray
        }

    def recognize_text(self, image):
        try:
            # 获取多种预处理后的图像
            processed_images = self.preprocess_image(image)

            best_text = ""
            best_method = ""

            # 尝试不同的预处理方法和OCR配置
            methods = [
                ("adaptive_eng", processed_images["adaptive"], r'--oem 3 --psm 6 -l eng'),
                ("otsu_eng", processed_images["otsu"], r'--oem 3 --psm 6 -l eng'),
                ("original_eng", processed_images["original_gray"], r'--oem 3 --psm 6 -l eng'),
                # 尝试中文识别（如果有中文语言数据）
                ("adaptive_chi_sim", processed_images["adaptive"], r'--oem 3 --psm 6 -l chi_sim'),
                ("otsu_chi_sim", processed_images["otsu"], r'--oem 3 --psm 6 -l chi_sim'),
            ]

            for method_name, processed_img, config_str in methods:
                try:
                    text = pytesseract.image_to_string(processed_img, config=config_str)
                    cleaned = ' '.join(text.strip().split())

                    # 如果识别到文字且比之前的好，就更新
                    if cleaned and len(cleaned) > len(best_text):
                        best_text = cleaned
                        best_method = method_name
                        print(f"方法 {method_name} 识别到 {len(cleaned)} 个字符: {cleaned[:50]}...")

                except Exception as e:
                    print(f"方法 {method_name} 失败: {e}")

            if best_text:
                self.input_text.delete("1.0", tk.END)
                self.input_text.insert(tk.END, best_text)
                self.status_var.set(f"识别完成（{best_method}）：{len(best_text)}字符")
                self.translate_text()
            else:
                # 如果没有识别到文字，尝试保存图像用于调试
                try:
                    debug_path = "debug_screenshot.png"
                    image.save(debug_path)
                    self.status_var.set(f"未识别到有效文字，已保存到 {debug_path}")
                    print(f"未识别到文字，截图已保存到: {debug_path}")
                except:
                    self.status_var.set("未识别到有效文字")

        except Exception as e:
            error_msg = f"OCR失败: {str(e)}"
            self.status_var.set(error_msg)
            print(error_msg)

    # --- 翻译核心 ---
    def translate_text(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            self.status_var.set("请输入要翻译的文本")
            return

        self.status_var.set("正在翻译...")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "🧠 思考中...\n")

        thread = threading.Thread(target=self._translate_worker, args=(text,))
        thread.daemon = True
        thread.start()

    def _translate_worker(self, text):
        try:
            clean_text = text.strip()
            if not clean_text:
                self.root.after(0, lambda: self._update_result("❌ 输入为空"))
                return

            # ✅ 安全编码：确保所有字符串为 UTF-8
            safe_system = SYSTEM_PROMPT.encode('utf-8', errors='replace').decode('utf-8')
            safe_prompt = (f"{PRE_PROMPT}\n{clean_text}").encode('utf-8', errors='replace').decode('utf-8')

            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": safe_system},
                    {"role": "user", "content": safe_prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=1024,
                stream=False
            )

            result = response.choices[0].message.content
            self.root.after(0, lambda: self._update_result(result))

        except Exception as e:
            # ✅ 安全处理错误信息
            error_str = str(e)
            try:
                safe_error = error_str.encode('utf-8', errors='replace').decode('utf-8')
            except:
                safe_error = "未知错误"
            error_msg = f"❌ API请求失败：{safe_error}"
            self.root.after(0, lambda: self._update_result(error_msg))

    def _update_result(self, result):
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, result)
        self.status_var.set("✅ 翻译完成")

    def clear_all(self):
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)
        self.preview_label.config(image="", text="无截图")
        self.status_var.set("已清空")


if __name__ == "__main__":
    app = GameTranslationAssistant()
    app.root.mainloop()