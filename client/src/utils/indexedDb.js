async function openDatabase() {
  if (!("indexedDB" in window)) {
    throw new Error("Browser does not support IndexedDB");
  }

  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open("FilesDatabase", 1);

    request.onupgradeneeded = function (event) {
      const db = event.target.result;
      db.createObjectStore("fileHandles", { keyPath: "id" });
    };

    request.onerror = function (event) {
      reject("Database error: " + event.target.errorCode);
    };

    request.onsuccess = function (event) {
      resolve(event.target.result);
    };
  });
}

export async function saveFileHandle(fileHandle) {
  const db = await openDatabase();
  const transaction = db.transaction("fileHandles", "readwrite");
  const store = transaction.objectStore("fileHandles");

  const request = store.put({
    id: fileHandle.name,
    handle: fileHandle,
    vector: null,
  });

  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function updateFileHandleVector(name, vector) {
  const db = await openDatabase();
  const transaction = db.transaction("fileHandles", "readwrite");
  const store = transaction.objectStore("fileHandles");

  const request = store.put({ id: name, vector });

  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function getFileHandle(id) {
  const db = await openDatabase();
  const transaction = db.transaction("fileHandles", "readonly");
  const store = transaction.objectStore("fileHandles");

  const request = store.get(id);

  return new Promise((resolve, reject) => {
    request.onsuccess = () => {
      if (request.result) {
        resolve(request.result.handle);
      } else {
        reject("No file handle found");
      }
    };
    request.onerror = () => reject(request.error);
  });
}

export async function clearObjectStore() {
  const db = await openDatabase();
  const transaction = db.transaction("fileHandles", "readwrite");
  const store = transaction.objectStore("fileHandles");

  return new Promise((resolve, reject) => {
    const request = store.clear();

    request.onsuccess = () => {
      console.log("Object store cleared");
      resolve();
    };
    request.onerror = () => {
      console.error("Error clearing object store:", request.error);
      reject(request.error);
    };
  });
}

export async function getAllFileHandles() {
  const db = await openDatabase();
  const transaction = db.transaction("fileHandles", "readonly");
  const store = transaction.objectStore("fileHandles");

  return new Promise((resolve, reject) => {
    const request = store.openCursor();
    const fileHandles = [];

    request.onsuccess = (event) => {
      const cursor = event.target.result;
      if (cursor) {
        fileHandles.push(cursor.value);
        cursor.continue();
      } else {
        resolve(fileHandles);
      }
    };

    request.onerror = () => {
      console.error("Failed to retrieve file handles:", request.error);
      reject(request.error);
    };
  });
}
