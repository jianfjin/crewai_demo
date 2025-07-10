import * as React from "react"
import { X } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

interface MultiSelectProps {
  options: string[]
  value: string[]
  onChange: (value: string[]) => void
  placeholder?: string
  className?: string
}

export function MultiSelect({
  options,
  value,
  onChange,
  placeholder = "Select items...",
  className
}: MultiSelectProps) {
  const [open, setOpen] = React.useState(false)

  const handleSelect = (selectedValue: string) => {
    if (!value.includes(selectedValue)) {
      onChange([...value, selectedValue])
    }
    setOpen(false)
  }

  const handleRemove = (valueToRemove: string) => {
    onChange(value.filter(item => item !== valueToRemove))
  }

  const availableOptions = options.filter(option => !value.includes(option))

  return (
    <div className={className}>
      <div className="flex flex-wrap gap-1 mb-2">
        {value.map((item) => (
          <Badge key={item} variant="secondary" className="text-xs">
            {item}
            <Button
              variant="ghost"
              size="sm"
              className="ml-1 h-auto p-0 text-muted-foreground hover:text-foreground"
              onClick={() => handleRemove(item)}
            >
              <X className="h-3 w-3" />
            </Button>
          </Badge>
        ))}
      </div>
      
      {availableOptions.length > 0 && (
        <Select open={open} onOpenChange={setOpen} onValueChange={handleSelect}>
          <SelectTrigger>
            <SelectValue placeholder={placeholder} />
          </SelectTrigger>
          <SelectContent>
            {availableOptions.map((option) => (
              <SelectItem key={option} value={option}>
                {option}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </div>
  )
}