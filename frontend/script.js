let position = [
  ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
  ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
  ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
]

const pieceMap = {
  'r': 'rook',
  'n': 'knight',
  'b': 'bishop',
  'q': 'queen',
  'k': 'king',
  'p': 'pawn'
}

let visibilityMask = [
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1]
]

let selectedPiece = null;
let playerSide = 'white';

function populateBoard() {
  var board = document.getElementById('board');
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      var square = document.createElement('div');
      square.className = 'cell';
      square.classList.add((i + j) % 2 === 0 ? 'light' : 'dark');
      square.addEventListener('click', squareClicked);
      board.appendChild(square);
    }
  }
}

function placePieces() {
  var board = document.getElementById('board');
  var squares = board.getElementsByClassName('cell');
  for (let i = 0; i < 64; i++) {
    var piece = document.createElement('img');
    if (position[Math.floor(i / 8)][i % 8] !== ' ') {
      const pieceText = position[Math.floor(i / 8)][i % 8];
      const pieceIsDark = pieceText.toLowerCase() === pieceText;
      piece.src = `assets/${pieceIsDark ? 'black' : 'white'}-${pieceMap[pieceText.toLowerCase()]}.png`;
      piece.className = 'piece';
    }
    squares[i].appendChild(piece);
  }
}

function updatePieces() {
  var board = document.getElementById('board');
  var squares = board.getElementsByClassName('cell');
  for (let i = 0; i < 64; i++) {
    if (squares[i].getElementsByClassName('piece').length === 0 && position[Math.floor(i / 8)][i % 8] !== ' ') {
      var piece = document.createElement('img');
      const pieceText = position[Math.floor(i / 8)][i % 8];
      const pieceIsDark = pieceText.toLowerCase() === pieceText;
      piece.src = `assets/${pieceIsDark ? 'black' : 'white'}-${pieceMap[pieceText.toLowerCase()]}.png`;
      piece.className = 'piece';
      squares[i].appendChild(piece);
    }
    else if (squares[i].getElementsByClassName('piece').length > 0 && position[Math.floor(i / 8)][i % 8] === ' ') {
      squares[i].removeChild(squares[i].getElementsByClassName('piece')[0]);
    } else if (squares[i].getElementsByClassName('piece').length > 0) {
      const piece = squares[i].getElementsByClassName('piece')[0];
      const pieceText = position[Math.floor(i / 8)][i % 8];
      const pieceIsDark = pieceText.toLowerCase() === pieceText;
      piece.src = `assets/${pieceIsDark ? 'black' : 'white'}-${pieceMap[pieceText.toLowerCase()]}.png`;
    }
  }
}

function updateVisibility() {
  var board = document.getElementById('board');
  var squares = board.getElementsByClassName('cell');
  for (let i = 0; i < 64; i++) {
    if (squares[i].getElementsByClassName('piece').length === 0) {
      continue;
    } 
    let piece = squares[i].getElementsByClassName('piece')[0];
    
    piece.setAttribute('style', `visibility: ${visibilityMask[Math.floor(i / 8)][i % 8] === 1 ? 'visible' : 'hidden'}`);
  }
}

function determineAllegienceSwitch() {
  if (Math.random() < 0.25) {
    let whitePieceIndices = [];
    for (let i = 0; i < 64; i++) {
      if (position[Math.floor(i / 8)][i % 8] !== ' ' && position[Math.floor(i / 8)][i % 8].toLowerCase() !== position[Math.floor(i / 8)][i % 8]) {
        whitePieceIndices.push(i);
      }
    }

    let randomIndex = whitePieceIndices[Math.floor(Math.random() * whitePieceIndices.length)];
    let piece = document.getElementsByClassName('cell')[randomIndex].getElementsByClassName('piece')[0];
    let pieceText = position[Math.floor(randomIndex / 8)][randomIndex % 8];

    position[Math.floor(randomIndex / 8)][randomIndex % 8] = pieceText.toLowerCase();
    updatePieces();
  }
}

function squareClicked() {
  const square = this;
  if (selectedPiece === null) {
    if (square.getElementsByClassName('piece').length === 0) {
      return;
    }

    const piece = square.getElementsByClassName('piece')[0];
    const pieceIsDark = piece.src.includes('black');
    if (pieceIsDark && playerSide === 'white' || !pieceIsDark && playerSide === 'black') {
      return;
    }

    selectedPiece = square;
    square.classList.add('selected');
  }
  else {
    if (selectedPiece === square) {
      selectedPiece = null;
      square.classList.remove('selected');
      return;
    }

    if (isValidMove(square, selectedPiece)) {
      // Update position array
      const piece = selectedPiece.getElementsByClassName('piece')[0];
      const pieceText = piece.src.split('/').pop().split('.')[0].split('-')[1];
      const pieceRow = Math.floor(Array.from(selectedPiece.parentNode.children).indexOf(selectedPiece) / 8);
      const pieceCol = Array.from(selectedPiece.parentNode.children).indexOf(selectedPiece) % 8;
      const squareRow = Math.floor(Array.from(square.parentNode.children).indexOf(square) / 8);
      const squareCol = Array.from(square.parentNode.children).indexOf(square) % 8;

      position[squareRow][squareCol] = position[pieceRow][pieceCol];
      position[pieceRow][pieceCol] = ' ';

      selectedPiece.classList.remove('selected');
      selectedPiece = null;

      updatePieces();

      fetch('/api/move/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Origin': 'https://github.com'
        },
        body: JSON.stringify({
          position: position,
        })
      })
        .then(response => response.json())
        .then(data => {
          let move = data.move;
          if (move === 'resign') {
            alert('You win (wattesigma)!');
          }

          let start = move.substring(0, 2);
          let end = move.substring(2, 4);

          let startCol = start.charCodeAt(0) - 97;
          let startRow = 8 - parseInt(start[1]);
          let endCol = end.charCodeAt(0) - 97;
          let endRow = 8 - parseInt(end[1]);

          position[endRow][endCol] = position[startRow][startCol];
          position[startRow][startCol] = ' ';

          updatePieces();
        });

      // Randomize Visibility mask
      visibilityMask = visibilityMask.map(row => row.map(() => Math.random() < 0.5 ? 0 : 1));
      updateVisibility();
      determineAllegienceSwitch();
      updateSmackTalk();
      changePlayerSide();
    }
  }
}

function isValidMove(square, selectedPiece) {
  const piece = selectedPiece.getElementsByClassName('piece')[0];

  const pieceText = piece.src.split('/').pop().split('.')[0].split('-')[1];

  const pieceIsDark = piece.src.includes('black');
  const pieceRow = Math.floor(Array.from(selectedPiece.parentNode.children).indexOf(selectedPiece) / 8);
  const pieceCol = Array.from(selectedPiece.parentNode.children).indexOf(selectedPiece) % 8;
  const squareRow = Math.floor(Array.from(square.parentNode.children).indexOf(square) / 8);
  const squareCol = Array.from(square.parentNode.children).indexOf(square) % 8;

  const squarePiece = square.getElementsByClassName('piece').length > 0 ? square.getElementsByClassName('piece')[0] : null;

  if (pieceText === 'pawn') {
    // black go down white go up asl rea
    let offset = playerSide === 'white' ? pieceIsDark ? 1 : -1 : pieceIsDark ? -1 : 1;

    let attackingSquareLeft = pieceCol !== 0 ? document.getElementsByClassName('cell')[8 * (pieceRow + offset) + pieceCol - 1] : null;
    let attackingSquareRight = pieceCol !== 7 ? document.getElementsByClassName('cell')[8 * (pieceRow + offset) + pieceCol + 1] : null;

    return squarePiece === null ? 
      squareRow === pieceRow + offset && squareCol === pieceCol : 
      (square === attackingSquareLeft || square === attackingSquareRight) && (squarePiece.src.includes(pieceIsDark ? 'white' : 'black'));
  } else if (pieceText === 'rook') {
    return (squareRow === pieceRow || squareCol === pieceCol) && 
      (squareRow !== pieceRow || squareCol !== pieceCol) && 
      !isPieceBetween(squareRow, squareCol, pieceRow, pieceCol) && 
      (squarePiece === null || squarePiece.src.includes(pieceIsDark ? 'white' : 'black'));
  } else if (pieceText === 'knight') {
    return (Math.abs(squareRow - pieceRow) === 2 && Math.abs(squareCol - pieceCol) === 1) || 
      (Math.abs(squareRow - pieceRow) === 1 && Math.abs(squareCol - pieceCol) === 2) && 
      (squarePiece === null || squarePiece.src.includes(pieceIsDark ? 'white' : 'black'));
  } else if (pieceText === 'bishop') {
    return Math.abs(squareRow - pieceRow) === Math.abs(squareCol - pieceCol) && 
      !isPieceBetween(squareRow, squareCol, pieceRow, pieceCol) && 
      (squarePiece === null || squarePiece.src.includes(pieceIsDark ? 'white' : 'black'));
  } else if (pieceText === 'queen') {
    return (squareRow === pieceRow || squareCol === pieceCol || Math.abs(squareRow - pieceRow) === Math.abs(squareCol - pieceCol)) && 
      (squareRow !== pieceRow || squareCol !== pieceCol) && 
      !isPieceBetween(squareRow, squareCol, pieceRow, pieceCol) && 
      (squarePiece === null || squarePiece.src.includes(pieceIsDark ? 'white' : 'black'));
  } else if (pieceText === 'king') {
    return Math.abs(squareRow - pieceRow) <= 1 && Math.abs(squareCol - pieceCol) <= 1 && 
      (squarePiece === null || squarePiece.src.includes(pieceIsDark ? 'white' : 'black'));
  }

  return false;
}

function isPieceBetween(row1, col1, row2, col2) {
  if (row1 === row2) {
    for (let i = Math.min(col1, col2) + 1; i < Math.max(col1, col2); i++) {
      if (document.getElementsByClassName('cell')[8 * row1 + i].getElementsByClassName('piece').length > 0) {
        return true;
      }
    }
  } else if (col1 === col2) {
    for (let i = Math.min(row1, row2) + 1; i < Math.max(row1, row2); i++) {
      if (document.getElementsByClassName('cell')[8 * i + col1].getElementsByClassName('piece').length > 0) {
        return true;
      }
    }
  }
  return false;
}

function updateSmackTalk() {
  let usuck = document.getElementById('usuck');
  fetch('/api/usuck')
    .then(response => response.json())
    .then(data => {
      usuck.innerText = data.message;
    });
}

function changePlayerSide() {
  if (Math.random() < 0.1) {
    position = position.map(row => row.map(piece => piece === ' ' ? ' ' : (piece.toLowerCase() === piece ? piece.toUpperCase() : piece.toLowerCase())));
    playerSide = playerSide === 'white' ? 'black' : 'white';
    fetch('/api/side', {method: 'PUT'})
    updatePieces();
  }
}

populateBoard();
placePieces();
