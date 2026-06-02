-- dash_sanitizer.lua
-- Rewrites every U+2014 (em-dash) and U+2013 (en-dash) in body text
-- to a hyphen surrounded by spaces. Math and Code elements are left
-- intact: equations might contain genuine minus signs typeset via en-dash
-- in some fonts.

local function fix_str(s)
  s = s:gsub("\u{2014}", " - ")   -- em-dash -> " - "
  s = s:gsub("\u{2013}", "-")     -- en-dash -> "-"
  return s
end

function Str(el)
  el.text = fix_str(el.text)
  return el
end

function RawInline(el)
  -- Preserve raw inlines (e.g., LaTeX math) untouched.
  return el
end

function Code(el)
  return el
end

function CodeBlock(el)
  return el
end

function Math(el)
  return el
end
