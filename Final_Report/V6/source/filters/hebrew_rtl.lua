-- hebrew_rtl.lua
-- Pandoc's LaTeX reader maps \begin{hebrew}...\end{hebrew} (polyglossia)
-- to a Div with lang="he" but no direction attribute, so the DOCX writer
-- emits LTR paragraphs and Hebrew text renders with broken punctuation
-- and word order (the V5 bidi defect). This filter adds dir="rtl" to any
-- element tagged with a Hebrew language code; the DOCX writer then sets
-- w:bidi on the paragraph properties.

local function make_rtl(el)
  local lang = el.attributes and el.attributes["lang"]
  if lang and lang:match("^he") then
    el.attributes["dir"] = "rtl"
    return el
  end
  return nil
end

return {
  { Div = make_rtl, Span = make_rtl },
}
