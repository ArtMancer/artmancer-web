# 🎨 Artmancer Web

## AI-Powered Art Generation Platform

Artmancer is a modern web application that enables users to generate stunning AI artwork through intuitive text prompts and image references. Built with Next.js 15 and featuring a sleek dark theme with purple accents.

![Artmancer Interface](https://img.shields.io/badge/Interface-Modern%20UI-6a0dad)
![Next.js](https://img.shields.io/badge/Next.js-15.5.3-black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue)
![Tailwind CSS](https://img.shields.io/badge/Tailwind%20CSS-4.x-38bdf8)

## ✨ Features

- 🎨 **AI Art Generation** - Create stunning artwork from text descriptions
- 📷 **Image Upload** - Use reference images for enhanced generation
- 🎛️ **Customization Panel** - Control style, quality, and generation parameters
- 📱 **Responsive Design** - Optimized for desktop and mobile devices
- 🌙 **Dark Theme** - Modern dark interface with purple accent colors
- 🔄 **Dynamic Sizing** - Multiple output resolutions (512x512, 768x768, 1024x1024)
- ⚡ **Fast Performance** - Built with Next.js 15 and Turbopack

## 🚀 Quick Start

### Prerequisites

Make sure you have the following installed:

- **Node.js** 18.0 or higher
- **npm** 8.0 or higher (or **yarn** 1.22+)
- **Git** for version control

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/nxank4/artmancer-web.git
   cd artmancer-web
   ```

2. **Install dependencies**

   ```bash
   cd client
   npm install
   ```

3. **Start the development server**

   ```bash
   npm run dev
   ```

4. **Open your browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

## 🛠️ Development

### Available Scripts

In the `client` directory, you can run:

- **`npm run dev`** - Starts the development server with Turbopack
- **`npm run build`** - Builds the app for production
- **`npm run start`** - Runs the built app in production mode

### Project Structure

```tree
artmancer-web/
├── client/                 # Next.js frontend application
│   ├── public/            # Static assets
│   │   ├── favicon.ico    # Custom favicon
│   │   ├── logo.svg       # Artmancer logo
│   │   └── *.svg          # UI icons
│   ├── src/
│   │   ├── app/           # Next.js App Router
│   │   │   ├── globals.css    # Global styles & theme
│   │   │   ├── layout.tsx     # Root layout
│   │   │   └── page.tsx       # Main application page
│   │   └── components/    # (Future component directory)
│   ├── package.json
│   └── next.config.ts
├── server/                # Backend API (Future implementation)
└── README.md
```

### Technology Stack

#### Frontend

- **Next.js 15** - React framework with App Router
- **React 19** - UI library
- **TypeScript 5** - Type safety
- **Tailwind CSS 4** - Utility-first CSS framework
- **Turbopack** - Fast bundler for development

#### Design System

- **Dark Theme** - Modern dark interface
- **Purple Accent** - Brand color scheme (#6a0dad)
- **Responsive Layout** - Mobile-first design
- **Dotted Background Pattern** - Subtle texture

## 🎨 Design Features

### Color Palette

- **Primary Background**: `#0b0b0d` (Deep dark)
- **Secondary Background**: `#1a1a1f` (Lighter dark)
- **Primary Accent**: `#6a0dad` (Purple)
- **Highlight Accent**: `#9d4edd` (Light purple)
- **Text Primary**: `#e0e0e0` (Light gray)
- **Text Secondary**: `#a0a0a8` (Medium gray)

### UI Components

- **Header**: Logo, search input, and action buttons
- **Art Display**: Dynamic square canvas with size options
- **Customize Panel**: Collapsible settings sidebar
- **Image Upload**: Drag-and-drop or click-to-upload

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file in the `client` directory:

```env
# API Configuration (Future)
NEXT_PUBLIC_API_URL=http://localhost:8000

# AI Service Configuration (Future)
OPENAI_API_KEY=your_openai_key_here
STABILITY_API_KEY=your_stability_key_here
```

### Custom Fonts

The application uses the following Google Fonts:

- **Geist Sans** - Primary interface font
- **Geist Mono** - Code and monospace text

## 📱 Browser Support

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Use TypeScript for all new code
- Follow the existing code style
- Add appropriate comments for complex logic
- Test your changes across different screen sizes
- Ensure accessibility standards are met

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**

   ```bash
   npx kill-port 3000
   npm run dev
   ```

2. **Node modules issues**

   ```bash
   rm -rf node_modules package-lock.json
   npm install
   ```

3. **Build errors**

   ```bash
   npm run build
   # Check console for specific errors
   ```

## 🚧 Roadmap

### Phase 1 (Current)

- ✅ Frontend UI/UX implementation
- ✅ Image upload functionality
- ✅ Responsive design
- ✅ Theme customization

### Phase 2 (Upcoming)

- 🔄 Backend API development
- 🔄 AI model integration
- 🔄 User authentication
- 🔄 Generation history

### Phase 3 (Future)

- 📋 Advanced editing tools
- 📋 Social features
- 📋 Premium subscriptions
- 📋 Mobile app

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Developer**: [nxank4](https://github.com/nxank4)
- **Project**: Artmancer Web Application

## 🙏 Acknowledgments

- [Next.js](https://nextjs.org/) for the amazing React framework
- [Tailwind CSS](https://tailwindcss.com/) for the utility-first CSS
- [Vercel](https://vercel.com/) for deployment platform
- The open-source community for inspiration and tools

---

### Built with ❤️ and ☕ by the Artmancer team

For questions or support, please open an issue on GitHub.
