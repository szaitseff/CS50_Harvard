// DRAWING ON CANVAS

var canvas = document.getElementById('canvas');

// Get the Canvas context and set its properties
var ctx = canvas.getContext('2d');
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 1;
ctx.setLineDash([4, 2]);
ctx.strokeRect(40, 40, 200, 200);

// Define the mouse object and onPaint method
var mouse = {x: 0, y: 0};
var onPaint = function() {
    ctx.lineTo(mouse.x, mouse.y);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 20;
    ctx.lineCap = 'butt';
    ctx.lineJoin = 'bevel';
    ctx.setLineDash([]);
    ctx.stroke();
};

// Actions on mouse events
canvas.addEventListener('mousemove', function(e) {
  mouse.x = e.pageX - this.offsetLeft;
  mouse.y = e.pageY - this.offsetTop;
}, false);

canvas.addEventListener('mousedown', function(e) {
    ctx.beginPath();
    ctx.moveTo(mouse.x, mouse.y);
    canvas.addEventListener('mousemove', onPaint, false);
}, false);

canvas.addEventListener('mouseup', function() {
    canvas.removeEventListener('mousemove', onPaint, false);
}, false);


// BUTTONS

// bind event handler to CLEAR button
$('#clear').click(function() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 2]);
    ctx.strokeRect(40, 40, 200, 200);
});

// bind event handler to "Ask Keras Model" button (POST request to /keras/)
$('#predict-keras').click(function() {
    // save canvas image as data url (png format by default)
    var dataURL = canvas.toDataURL();
    $.ajax({
        type: 'POST',
        url: '/keras/',
        data: dataURL,
        success: function(response) {
            $('#result').text('Predicted Output:  ' + response);
        }
    });
});

// bind event handler to "Ask Custom Model" button (POST request to /custom/)
$('#predict-custom').click(function() {
    // save canvas image as data url (png format by default)
    var dataURL = canvas.toDataURL();
    $.ajax({
        type: 'POST',
        url: '/custom/',
        data: dataURL,
        success: function(response) {
            $('#result').text('Predicted Output:  ' + response);
        }
    });
});
