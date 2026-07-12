/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#0e1117',
        panel: '#161b22',
        line: '#232a34',
        accent: '#4f8cff',
      },
    },
  },
  plugins: [],
}
