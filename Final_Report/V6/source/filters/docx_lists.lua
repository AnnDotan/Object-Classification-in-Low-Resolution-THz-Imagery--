-- docx_lists.lua  (V6)
-- Two DOCX-deliverable fixes that the plain Pandoc LaTeX->DOCX path lacks:
--
-- 1. Caption numbering. Pandoc resolves \ref{...} to "Figure 4"-style link
--    text but writes the caption paragraphs themselves UNNUMBERED (the V5
--    defect: prose says "Figure 4" while no caption says which figure is
--    which). This filter walks the document in order and prepends
--    "Figure N: " / "Table N: " to every Figure / Table caption, matching
--    Pandoc's own \ref numbering (both count in document order).
--
-- 2. Native Word list fields. The three unnumbered section headings
--    "Table of Contents" / "List of Figures" / "List of Tables" are
--    restyled as TOCHeading (so they do not list themselves) and followed
--    by a Word TOC field: heading outline for the TOC, caption-style
--    harvest for the LOF / LOT. The fields are marked dirty so Word offers
--    to populate them on first open (or Ctrl+A, F9).

local FIELD_FMT = [[
<w:p>
  <w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>
  <w:r><w:instrText xml:space="preserve"> %s </w:instrText></w:r>
  <w:r><w:fldChar w:fldCharType="separate"/></w:r>
  <w:r><w:t xml:space="preserve">%s</w:t></w:r>
  <w:r><w:fldChar w:fldCharType="end"/></w:r>
</w:p>
]]

local HEADING_FMT = [[
<w:p>
  <w:pPr><w:pStyle w:val="TOCHeading"/></w:pPr>
  <w:r><w:t xml:space="preserve">%s</w:t></w:r>
</w:p>
]]

local PLACEHOLDER = "Right-click and choose Update Field (or press Ctrl+A then F9) to populate this list."

local LIST_SPECS = {
  ["Table of Contents"] = 'TOC \\o "1-2" \\h \\z \\u',
  ["List of Figures"]   = 'TOC \\h \\z \\t "Image Caption,1"',
  ["List of Tables"]    = 'TOC \\h \\z \\t "Table Caption,1"',
}

local function stringify_header(h)
  return pandoc.utils.stringify(h.content)
end

function Pandoc(doc)
  local fig_n, tab_n = 0, 0

  doc = doc:walk({
    traverse = "topdown",
    -- Count ONLY captioned floats: Pandoc's \ref resolver numbers captioned
    -- figures/tables in document order and skips captionless layout floats
    -- (e.g., the cover-page tabulars). Counting those too would shift every
    -- caption number away from the in-text references.
    Figure = function(fig)
      local cap = fig.caption.long
      if cap and #cap > 0 and (cap[1].t == "Plain" or cap[1].t == "Para") then
        fig_n = fig_n + 1
        local prefix = {
          pandoc.Strong({ pandoc.Str("Figure " .. fig_n .. ":") }),
          pandoc.Space(),
        }
        for i = #prefix, 1, -1 do
          table.insert(cap[1].content, 1, prefix[i])
        end
      end
      return fig
    end,
    Table = function(tbl)
      local cap = tbl.caption.long
      if cap and #cap > 0 and (cap[1].t == "Plain" or cap[1].t == "Para") then
        tab_n = tab_n + 1
        local prefix = {
          pandoc.Strong({ pandoc.Str("Table " .. tab_n .. ":") }),
          pandoc.Space(),
        }
        for i = #prefix, 1, -1 do
          table.insert(cap[1].content, 1, prefix[i])
        end
      end
      return tbl
    end,
  })

  local out = pandoc.List()
  for _, blk in ipairs(doc.blocks) do
    local handled = false
    if blk.t == "Header" and blk.level == 1 then
      local title = stringify_header(blk)
      local instr = LIST_SPECS[title]
      if instr then
        out:insert(pandoc.RawBlock("openxml", string.format(HEADING_FMT, title)))
        out:insert(pandoc.RawBlock("openxml",
                                   string.format(FIELD_FMT, instr, PLACEHOLDER)))
        handled = true
      end
    end
    if not handled then
      out:insert(blk)
    end
  end
  doc.blocks = out
  return doc
end
