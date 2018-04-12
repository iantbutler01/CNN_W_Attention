var fileJSON = ''
var reading = false
const fileUploadHandler = (event) => {
  event.preventDefault()
  reading = true
  for(x of event.dataTransfer.items) {
    if (x.kind === 'file') {
      let file = x.getAsFile();
      let reader = new FileReader()
      reader.onload = (e) => {
        fileJSON = JSON.stringify(reader.result)
        reading = false
      }
      reader.onerror = (e) => {
        reading = false
      }
      reader.readAsText(file)
    }
  }
}

const trainClickHandler = (event) => {
  if(reading) {
    return
  }
  fetch('/train', { body: fileJSON, method: "POST", headers: { 'Content-Type': 'application/json' } }).then((res) => {
    let elem = document.getElementById('idBox')
    res.json().then((data) =>  {
      elem.innerText = data['model_id']
      elem.style.display = 'block'
    })
  })
}

const predictClickHandler = (event) => {
  if(reading) {
    return
  }
  fetch('/predict', { body: fileJSON, method: "POST", headers: { 'Content-Type': 'application/json' } }).then((res) => {
    let elem = document.getElementById('predBox')
    res.json().then((data) =>  {
      elem.innerText = data['prediction']
      elem.style.display = 'block'
    })
  })

}

const dragHandler = (event) => {
  event.preventDefault()
}
