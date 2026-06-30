/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"DM Sans"', 'system-ui', 'sans-serif'],
        display: ['"Fraunces"', 'Georgia', 'serif'],
      },
      colors: {
        ink: {
          950: '#0a0a0f',
          900: '#12121a',
          800: '#1a1a26',
          700: '#252533',
          600: '#3a3a4d',
          500: '#6b6b80',
          400: '#9898ad',
          300: '#c4c4d4',
          200: '#e8e8f0',
        },
        accent: {
          DEFAULT: '#7c6cff',
          light: '#9d8fff',
          glow: '#5b4fd9',
        },
        mint: '#3ee8b5',
      },
      animation: {
        'fade-up': 'fadeUp 0.5s ease-out forwards',
        'pulse-soft': 'pulseSoft 2s ease-in-out infinite',
        'typing': 'typing 1.4s ease-in-out infinite',
      },
      keyframes: {
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '0.4' },
          '50%': { opacity: '1' },
        },
        typing: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-4px)' },
        },
      },
      boxShadow: {
        glow: '0 0 40px -10px rgba(124, 108, 255, 0.45)',
        card: '0 4px 24px -4px rgba(0, 0, 0, 0.4)',
      },
    },
  },
  plugins: [],
}
